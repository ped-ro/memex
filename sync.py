#!/usr/bin/env python3
"""
Vault Incremental Sync Script
Checks vault for notes modified since last sync and re-indexes them.
Uses content hashing (SHA-256) for reliable change detection.
Auto-recovers from empty DB by treating all notes as new.

Usage:
    python3 sync.py [--vault /path/to/vault] [--full]
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime, timezone

# Reuse helpers from import.py
sys.path.insert(0, str(Path(__file__).parent))
from vault_import import (
    scan_vault, chunk_text, embed_texts, token_count, content_hash,
    get_connection, get_db_hashes, resolve_wikilinks,
    VAULT_PATH, DB_HOST, DB_PORT, DB_NAME
)

import psycopg2.extras

EMBED_BATCH = 32   # match vault_import.py ??? mxbai-large OOMs on M1 at 100


# ── Safety ────────────────────────────────────────────────────────────────────

def vault_accessible(vault_root: Path) -> bool:
    """Quick sanity check — can we read the vault?"""
    if not vault_root.exists():
        return False
    # Spot-check known files
    return (vault_root / "README.md").exists() or (vault_root / "CLAUDE.md").exists()


def check_sanity(vault_root: Path, vault_count: int, db_count: int) -> bool:
    """
    Before deleting anything, verify the vault looks real.
    Aborts if vault note count < 50% of DB count (vault likely broken/unmounted).
    """
    if db_count == 0:
        return True  # empty DB, nothing to protect

    if db_count > 10 and vault_count < db_count * 0.5:
        print(
            f"🚨 ABORT: Vault appears empty or broken — "
            f"{vault_count} .md files on disk vs {db_count} in DB. "
            f"Protecting index. Check that {vault_root} is accessible."
        )
        return False

    return True


# ── DB helpers ────────────────────────────────────────────────────────────────

def delete_note(conn, path: str):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM notes WHERE path = %s", (path,))


def upsert_note(conn, note: dict):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO notes (path, title, tags, note_type, note_date, created_at, modified_at,
                               frontmatter, wikilinks, word_count, content_hash, indexed_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, now())
            ON CONFLICT (path) DO UPDATE SET
                title        = EXCLUDED.title,
                tags         = EXCLUDED.tags,
                note_type    = EXCLUDED.note_type,
                note_date    = EXCLUDED.note_date,
                modified_at  = EXCLUDED.modified_at,
                frontmatter  = EXCLUDED.frontmatter,
                wikilinks    = EXCLUDED.wikilinks,
                word_count   = EXCLUDED.word_count,
                content_hash = EXCLUDED.content_hash,
                indexed_at   = now()
        """, (
            note["path"], note["title"],
            note["tags"] or [],
            note.get("note_type"),
            note.get("note_date"),
            note.get("created_at"),
            note.get("modified_at"),
            note["frontmatter"],
            note["wikilinks"] or [],
            note["word_count"],
            note["content_hash"],
        ))


def upsert_chunks(conn, note: dict):
    """Chunk, embed, and insert/update chunks for a single note."""
    chunks = chunk_text(note["content"])
    chunk_objs = [
        {"note_path": note["path"], "chunk_index": i,
         "content": c, "token_count": token_count(c)}
        for i, c in enumerate(chunks)
    ]

    # Embed via local service
    texts = [c["content"] for c in chunk_objs]
    embeddings = embed_texts(texts)
    for obj, emb in zip(chunk_objs, embeddings):
        obj["embedding"] = emb

    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE note_path = %s", (note["path"],))
        psycopg2.extras.execute_batch(cur, """
            INSERT INTO chunks (note_path, chunk_index, content, token_count, embedding)
            VALUES (%s, %s, %s, %s, %s)
        """, [
            (c["note_path"], c["chunk_index"], c["content"],
             c["token_count"], str(c["embedding"]))
            for c in chunk_objs
        ])

    return len(chunk_objs)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Incremental vault sync")
    parser.add_argument("--vault", default=str(VAULT_PATH))
    parser.add_argument("--full", action="store_true", help="Force re-embed all notes (still skips unchanged via hash)")
    args = parser.parse_args()

    vault_root = Path(args.vault)

    if not vault_accessible(vault_root):
        print(f"🚨 ABORT: Vault not accessible at {vault_root} — skipping sync")
        sys.exit(0)

    conn = get_connection()
    t_start = time.time()

    # Get current DB state
    db_hashes = get_db_hashes(conn)
    db_paths = set(db_hashes.keys())

    # Scan vault
    all_notes = scan_vault(vault_root)
    vault_paths = {n["path"] for n in all_notes}

    # Safety check
    if not check_sanity(vault_root, len(all_notes), len(db_paths)):
        conn.close()
        sys.exit(0)

    # ── Determine what needs work (hash-based) ────────────────────────────────
    deleted_paths = db_paths - vault_paths
    new_notes = [n for n in all_notes if n["path"] not in db_paths]
    changed_notes = [
        n for n in all_notes
        if n["path"] in db_paths and db_hashes.get(n["path"]) != n["content_hash"]
    ]

    # --full: also re-embed notes that have no hash yet (legacy rows) or
    # notes where hash matches but we want to force re-embed
    if args.full:
        # Include notes with NULL hash (pre-hash-era rows)
        legacy_notes = [
            n for n in all_notes
            if n["path"] in db_paths and db_hashes.get(n["path"]) is None
        ]
        changed_notes = changed_notes + legacy_notes

    needs_embedding = new_notes + changed_notes
    unchanged_count = len(all_notes) - len(new_notes) - len(changed_notes)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Vault sync — "
          f"{len(new_notes)} new, {len(changed_notes)} changed, "
          f"{unchanged_count} unchanged, {len(deleted_paths)} deleted")

    if not needs_embedding and not deleted_paths:
        print("  Nothing to do.")
        duration_ms = int((time.time() - t_start) * 1000)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO sync_log (notes_added, notes_updated, notes_deleted, chunks_total, duration_ms) "
                "VALUES (0,0,0,0,%s)", (duration_ms,))
        conn.commit()
        conn.close()
        return

    # ── Delete removed notes ──────────────────────────────────────────────────
    for path in deleted_paths:
        delete_note(conn, path)
        print(f"  🗑️  {path}")

    # ── Upsert metadata for ALL notes (cheap, keeps tags/titles fresh) ────────
    for note in all_notes:
        upsert_note(conn, note)
    conn.commit()

    # ── Chunk and embed only new/changed notes ────────────────────────────────
    total_chunks = 0
    for i, note in enumerate(needs_embedding, 1):
        tag = "➕" if note["path"] not in db_paths else "🔄"
        try:
            n_chunks = upsert_chunks(conn, note)
            total_chunks += n_chunks
            print(f"  {tag} {note['path']} ({n_chunks} chunks)")
        except Exception as e:
            print(f"  ⚠️  Error embedding {note['path']}: {e}")
            # Commit what we have so far so we don't lose progress
            conn.commit()
            continue

    conn.commit()

    # ── Refresh wiki-links ────────────────────────────────────────────────────
    resolve_wikilinks(conn, all_notes)

    duration_ms = int((time.time() - t_start) * 1000)
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO sync_log (notes_added, notes_updated, notes_deleted, chunks_total, duration_ms)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            len(new_notes),
            len(changed_notes),
            len(deleted_paths),
            total_chunks,
            duration_ms,
        ))
    conn.commit()
    conn.close()
    print(f"  ✅ Done in {duration_ms/1000:.1f}s")


if __name__ == "__main__":
    main()
