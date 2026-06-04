#!/usr/bin/env python3
"""
Vault Bulk Import Script
Reads Obsidian vault → chunks → embeds via local embeddings service → loads into pgvector

Changes from content are detected via SHA-256 hash — unchanged notes are skipped entirely.
No data is ever truncated unless --reset is explicitly passed.

Usage:
    python3 vault_import.py [--vault /path/to/vault] [--reset] [--dry-run]

Requirements:
    pip3 install psycopg2-binary tiktoken python-frontmatter requests
"""

import os
import sys
import re
import json
import hashlib
import argparse
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import frontmatter
import tiktoken
import psycopg2
import psycopg2.extras
import requests

# ── Config ─────────────────────────────────────────────────────────────────────

DB_HOST     = os.environ.get("PGHOST", "localhost")
DB_PORT     = int(os.environ.get("PGPORT", "5433"))
DB_NAME     = os.environ.get("PGDATABASE", "obsidian")
DB_USER     = os.environ.get("PGUSER", "postgres")
DB_PASS     = os.environ.get("PGPASSWORD", "")

VAULT_PATH       = Path(os.environ.get("VAULT_PATH", "/Volumes/home/Obsidian"))
_raw_embed_url = os.environ.get("EMBED_URL", "http://localhost:8765/embed")
EMBEDDINGS_URL   = os.environ.get("EMBEDDINGS_URL", _raw_embed_url.rsplit("/embed", 1)[0] if _raw_embed_url.endswith("/embed") else _raw_embed_url)
EMBEDDINGS_MODEL = os.environ.get("EMBEDDINGS_MODEL", "mixedbread-ai/mxbai-embed-large-v1")
EMBEDDINGS_DIM   = int(os.environ.get("EMBEDDINGS_DIM", "1024"))

CHUNK_TOKENS    = 512       # target tokens per chunk
CHUNK_OVERLAP   = 50        # overlap tokens between chunks
EMBED_BATCH     = 32        # chunks per embedding API call
SKIP_DIRS       = {".git", ".obsidian", ".trash", ".semantic-search"}
SKIP_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".pdf", ".mp3", ".mp4", ".zip"}

# Notes whose relative path contains any of these substrings are skipped entirely.
# Use this to exclude credential-bearing or sensitive notes from the search index.
SKIP_PATH_SUBSTRINGS = {
    "mcp auth",          # contains API keys (case-insensitive match below)
    "/credentials/",     # folder match
    "/secrets/",
    " api key",          # filename pattern
    " client secret",
    " api token",
}

# ── Tokenizer ──────────────────────────────────────────────────────────────────

enc = tiktoken.get_encoding("cl100k_base")

def token_count(text: str) -> int:
    return len(enc.encode(text))

# ── Hashing ────────────────────────────────────────────────────────────────────

def content_hash(text: str) -> str:
    """SHA-256 hash of note content. Used to skip re-embedding unchanged notes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_tokens: int = CHUNK_TOKENS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping token-based chunks."""
    tokens = enc.encode(text)
    if len(tokens) <= chunk_tokens:
        return [text]

    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_tokens
        chunk_tokens_slice = tokens[start:end]
        chunk_text_str = enc.decode(chunk_tokens_slice)
        chunks.append(chunk_text_str)
        if end >= len(tokens):
            break
        start = end - overlap

    return chunks

# ── Wiki-link extraction ───────────────────────────────────────────────────────

WIKILINK_RE = re.compile(r'\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]')

def extract_wikilinks(content: str) -> list[str]:
    """Extract [[wikilink]] targets from markdown content."""
    links = WIKILINK_RE.findall(content)
    return list(set(link.strip() for link in links))

# ── Note parsing ──────────────────────────────────────────────────────────────

def parse_note(path: Path, vault_root: Path) -> Optional[dict]:
    """Parse a markdown file into a note dict."""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"  ⚠️  Read error {path}: {e}")
        return None

    try:
        post = frontmatter.loads(raw)
    except Exception:
        post = frontmatter.Post(raw)

    meta = post.metadata
    content_str = post.content.strip()

    if not content_str:
        return None

    rel_path = str(path.relative_to(vault_root))

    # Extract title: H1 > frontmatter title > filename
    h1 = re.search(r'^#\s+(.+)$', content_str, re.MULTILINE)
    title = (
        h1.group(1).strip() if h1
        else meta.get("title", path.stem)
    )

    # Normalize tags
    tags = meta.get("tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",")]
    tags = [str(t).strip() for t in tags if t]

    # Date
    note_date = meta.get("date")
    if note_date and not isinstance(note_date, str):
        note_date = str(note_date)

    stat = path.stat()
    created_at  = datetime.fromtimestamp(stat.st_birthtime, tz=timezone.utc) if hasattr(stat, "st_birthtime") else None
    modified_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

    wikilinks = extract_wikilinks(raw)

    return {
        "path":         rel_path,
        "title":        title,
        "tags":         tags,
        "note_type":    meta.get("type"),
        "note_date":    note_date,
        "created_at":   created_at,
        "modified_at":  modified_at,
        "frontmatter":  json.dumps(dict(meta), default=str),
        "wikilinks":    wikilinks,
        "word_count":   len(content_str.split()),
        "content":      content_str,
        "content_hash": content_hash(content_str),
    }

# ── Vault scanner ──────────────────────────────────────────────────────────────

def scan_vault(vault_root: Path) -> list[dict]:
    """Walk vault directory and parse all markdown files."""
    notes = []
    for md_file in sorted(vault_root.rglob("*.md")):
        # Skip hidden dirs and known skip dirs
        parts = set(md_file.parts)
        if any(d in parts for d in SKIP_DIRS):
            continue
        if any(part.startswith(".") for part in md_file.relative_to(vault_root).parts[:-1]):
            continue
        # Skip notes with sensitive content (credentials, API keys)
        rel_str = str(md_file.relative_to(vault_root))
        if any(pat in rel_str.lower() for pat in SKIP_PATH_SUBSTRINGS):
            continue

        note = parse_note(md_file, vault_root)
        if note:
            notes.append(note)

    return notes

# ── Embedding ──────────────────────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using local embeddings service."""
    response = requests.post(
        f"{EMBEDDINGS_URL}/embed",
        json={"texts": texts, "normalize": True},
        timeout=300
    )
    response.raise_for_status()
    data = response.json()
    embeddings = data["embeddings"]
    if embeddings:
        dim = len(embeddings[0])
        if dim != EMBEDDINGS_DIM:
            raise RuntimeError(
                f"Embedding dimension mismatch: service returned {dim}, expected {EMBEDDINGS_DIM}"
            )
    return embeddings

# ── DB helpers ──────────────────────────────────────────────────────────────────

def get_connection():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASS,
        cursor_factory=psycopg2.extras.RealDictCursor,
    )

def get_db_hashes(conn) -> dict:
    """Get {path: content_hash} for all notes in DB."""
    with conn.cursor() as cur:
        cur.execute("SELECT path, content_hash FROM notes")
        return {row["path"]: row["content_hash"] for row in cur.fetchall()}

def resolve_wikilinks(conn, notes: list[dict]):
    """Resolve wikilink titles to actual note paths."""
    title_to_path = {}
    with conn.cursor() as cur:
        cur.execute("SELECT path, title FROM notes")
        for row in cur.fetchall():
            title_to_path[row["title"].lower()] = row["path"]
            stem = Path(row["path"]).stem.lower()
            title_to_path[stem] = row["path"]

    with conn.cursor() as cur:
        cur.execute("DELETE FROM links")
        for note in notes:
            for link_title in note.get("wikilinks", []):
                target_path = title_to_path.get(link_title.lower())
                cur.execute(
                    """INSERT INTO links (source_path, target_path, target_title)
                       VALUES (%s, %s, %s)
                       ON CONFLICT DO NOTHING""",
                    (note["path"], target_path, link_title)
                )
    conn.commit()
    print(f"  ✅ Wiki-links resolved and inserted")

# ── Main import ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Import Obsidian vault into pgvector")
    parser.add_argument("--vault", default=str(VAULT_PATH), help="Path to Obsidian vault")
    parser.add_argument("--reset", action="store_true", help="DANGEROUS: Delete all data before import")
    parser.add_argument("--dry-run", action="store_true", help="Parse and chunk only, no DB writes")
    parser.add_argument("--skip-embed", action="store_true", help="Insert chunks without embeddings")
    args = parser.parse_args()

    vault_root = Path(args.vault)
    if not vault_root.exists():
        print(f"❌ Vault not found: {vault_root}")
        sys.exit(1)

    # Test embeddings service
    if not args.skip_embed and not args.dry_run:
        try:
            health = requests.get(f"{EMBEDDINGS_URL}/health", timeout=10)
            health.raise_for_status()
            info = health.json()
            print(f"✅ Embeddings service: {info.get('model', 'unknown')} ({info.get('dim', '?')}d)")
        except Exception as e:
            print(f"❌ Embeddings service not available at {EMBEDDINGS_URL}: {e}")
            sys.exit(1)

    t_start = time.time()
    print(f"\n🔍 Scanning vault: {vault_root}")
    notes = scan_vault(vault_root)
    print(f"   Found {len(notes)} notes")

    if args.dry_run:
        total_chunks = sum(len(chunk_text(n["content"])) for n in notes)
        total_tokens = sum(token_count(n["content"]) for n in notes)
        print(f"\n📊 Dry run stats:")
        print(f"   Notes: {len(notes)}")
        print(f"   Total chunks: {total_chunks}")
        print(f"   Total tokens: {total_tokens:,}")
        return

    conn = get_connection()
    print(f"✅ Connected to pgvector at {DB_HOST}:{DB_PORT}")

    # ── Reset (explicit only) ─────────────────────────────────────────────────
    if args.reset:
        print("🗑️  --reset: Truncating all tables...")
        with conn.cursor() as cur:
            cur.execute("TRUNCATE chunks, links, notes, sync_log RESTART IDENTITY CASCADE")
        conn.commit()

    # ── Determine what changed ────────────────────────────────────────────────
    db_hashes = get_db_hashes(conn)
    vault_paths = {n["path"] for n in notes}
    db_paths = set(db_hashes.keys())

    deleted_paths = db_paths - vault_paths
    new_notes = [n for n in notes if n["path"] not in db_paths]
    changed_notes = [
        n for n in notes
        if n["path"] in db_paths and db_hashes.get(n["path"]) != n["content_hash"]
    ]
    unchanged_notes = [
        n for n in notes
        if n["path"] in db_paths and db_hashes.get(n["path"]) == n["content_hash"]
    ]

    print(f"\n📊 Diff: {len(new_notes)} new, {len(changed_notes)} changed, "
          f"{len(unchanged_notes)} unchanged, {len(deleted_paths)} deleted")

    # ── Delete removed notes ──────────────────────────────────────────────────
    if deleted_paths:
        print(f"\n🗑️  Deleting {len(deleted_paths)} removed notes...")
        with conn.cursor() as cur:
            for path in deleted_paths:
                cur.execute("DELETE FROM notes WHERE path = %s", (path,))
                print(f"  🗑️  {path}")
        conn.commit()

    # ── Upsert metadata for ALL notes (fast, no embedding) ────────────────────
    # Even unchanged notes get metadata refreshed (tags, title, mtime, etc.)
    print(f"\n📝 Upserting metadata for {len(notes)} notes...")
    with conn.cursor() as cur:
        for note in notes:
            cur.execute("""
                INSERT INTO notes (path, title, tags, note_type, note_date, created_at,
                                   modified_at, frontmatter, wikilinks, word_count,
                                   content_hash, indexed_at)
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
    conn.commit()
    print(f"   ✅ Metadata updated")

    # ── Chunk and embed only new/changed notes ────────────────────────────────
    needs_embedding = new_notes + changed_notes
    if not needs_embedding:
        print(f"\n✅ No content changes — skipping embedding")
    else:
        print(f"\n✂️  Chunking {len(needs_embedding)} notes...")
        all_chunks = []
        for note in needs_embedding:
            chunks = chunk_text(note["content"])
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "note_path":   note["path"],
                    "chunk_index": i,
                    "content":     chunk,
                    "token_count": token_count(chunk),
                })

        total_chunks = len(all_chunks)
        total_tokens = sum(c["token_count"] for c in all_chunks)
        print(f"   {total_chunks} chunks | {total_tokens:,} tokens")

        if not args.skip_embed:
            print(f"\n🔢 Embedding {len(all_chunks)} chunks in batches of {EMBED_BATCH}...")
            for i in range(0, len(all_chunks), EMBED_BATCH):
                batch = all_chunks[i:i + EMBED_BATCH]
                texts = [c["content"] for c in batch]
                try:
                    embeddings = embed_texts(texts)
                    for chunk, emb in zip(batch, embeddings):
                        chunk["embedding"] = emb
                    progress = min(i + EMBED_BATCH, len(all_chunks))
                    print(f"   {progress}/{len(all_chunks)} chunks embedded...", flush=True)
                except Exception as e:
                    print(f"\n  ⚠️  Embedding error at batch {i}: {e}")
                    raise
            print(f"\n   ✅ All chunks embedded")
        else:
            print("   ⏭️  Skipping embeddings (--skip-embed)")
            for chunk in all_chunks:
                chunk["embedding"] = None

        # ── Insert chunks (only for changed notes) ───────────────────────────
        print(f"\n💾 Inserting chunks...")
        with conn.cursor() as cur:
            # Delete old chunks for notes being re-indexed
            cur.execute("DELETE FROM chunks WHERE note_path = ANY(%s)",
                        ([n["path"] for n in needs_embedding],))

            psycopg2.extras.execute_batch(cur, """
                INSERT INTO chunks (note_path, chunk_index, content, token_count, embedding)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (note_path, chunk_index) DO UPDATE SET
                    content     = EXCLUDED.content,
                    token_count = EXCLUDED.token_count,
                    embedding   = EXCLUDED.embedding
            """, [
                (c["note_path"], c["chunk_index"], c["content"], c["token_count"],
                 str(c["embedding"]) if c["embedding"] else None)
                for c in all_chunks
            ], page_size=500)
        conn.commit()
        print(f"   ✅ {len(all_chunks)} chunks inserted")

    # ── Resolve wiki-links ────────────────────────────────────────────────────
    print(f"\n🔗 Resolving wiki-links...")
    resolve_wikilinks(conn, notes)

    # ── Sync log ────────────────────────────────────────────────────────────────
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
