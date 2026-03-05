#!/usr/bin/env python3
"""
Vault Sync Daemon — Docker service
Runs incremental sync every 15 minutes.
Vault is mounted as /vault inside the container (host: /Volumes/home/Obsidian).
"""

import os
import sys
import time
import logging
import subprocess
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [sync] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

VAULT_PATH   = Path(os.environ.get("VAULT_PATH", "/vault"))
SYNC_SCRIPT  = Path("/app/host/sync.py")
INTERVAL_SEC = int(os.environ.get("SYNC_INTERVAL_SEC", "900"))  # 15 min default
EMBED_URL    = os.environ.get("EMBED_URL", "http://embeddings:8765/embed")


def vault_ok() -> bool:
    """Quick check — is the vault mount alive and readable?"""
    try:
        return (VAULT_PATH / "README.md").exists() or (VAULT_PATH / "CLAUDE.md").exists()
    except OSError:
        return False


def run_sync():
    if not vault_ok():
        log.warning(f"Vault not accessible at {VAULT_PATH} — skipping sync")
        return

    log.info(f"Running sync...")
    env = {
        **os.environ,
        "VAULT_PATH":   str(VAULT_PATH),
        "EMBED_URL":    EMBED_URL,
        "PGHOST":       os.environ.get("PGHOST", "pgvector"),
        "PGPORT":       os.environ.get("PGPORT", "5432"),
        "PGDATABASE":   os.environ.get("PGDATABASE", "obsidian"),
        "PGUSER":       os.environ.get("PGUSER", "postgres"),
        "PGPASSWORD":   os.environ.get("PGPASSWORD", ""),
    }
    result = subprocess.run(
        [sys.executable, str(SYNC_SCRIPT), "--vault", str(VAULT_PATH)],
        env=env
    )
    if result.returncode != 0:
        log.warning(f"Sync exited with code {result.returncode}")
    else:
        log.info("Sync complete")


def main():
    log.info(f"Vault sync daemon starting")
    log.info(f"Vault: {VAULT_PATH} | Interval: {INTERVAL_SEC}s | Embed: {EMBED_URL}")

    # Wait for embeddings service to be ready
    log.info("Waiting for embeddings service...")
    for _ in range(30):
        try:
            import urllib.request
            with urllib.request.urlopen(
                EMBED_URL.replace("/embed", "/health"), timeout=3
            ) as r:
                if r.status == 200:
                    log.info("Embeddings service ready")
                    break
        except Exception:
            pass
        time.sleep(5)

    # Run immediately on startup
    run_sync()

    # Then on interval
    while True:
        time.sleep(INTERVAL_SEC)
        run_sync()


if __name__ == "__main__":
    main()
