# memex

A personal knowledge engine. Semantic search, MCP server, and sync stack for your second brain — exposed to Claude, ChatGPT, and any HTTP client.

**No cloud. No subscriptions. Everything runs on your machine.**

## What It Does

- Hybrid search combining vector similarity + full-text (RRF fusion) over your vault
- HyDE query expansion for short/conversational queries
- Time-decay scoring to surface recent notes
- 7 MCP tools: `search_vault`, `get_note`, `get_backlinks`, `related_notes`, `search_by_tag`, `recent_notes`, `vault_stats`
- REST endpoint for non-MCP clients (n8n, scripts, ChatGPT connectors)
- Incremental sync every 15 minutes — only re-embeds changed notes
- API key auth (Bearer, X-API-Key header, or query param)

## Architecture

```
Obsidian Vault (local)
    ↓ (mount)
sync daemon  →  embeddings service (mxbai-embed-large-v1, local)
    ↓
pgvector (PostgreSQL 17 + pgvector extension)
    ↓
MCP server (Node.js, port 3456)
    ├── Claude Desktop / Claude Code (MCP transport)
    ├── ChatGPT connector (REST via Cloudflare Tunnel)
    └── Any HTTP client
```

## Requirements

- Docker + Docker Compose
- ~4GB disk for the embedding model (downloaded on first build)
- ~500MB RAM for the embeddings service at idle
- Your notes accessible as a local directory (Obsidian, Markdown files, etc.)

## Quick Start

```bash
git clone https://github.com/ped-ro/memex
cd memex

# 1. Configure
cp .env.example .env
# Edit .env — set VAULT_PATH, PGPASSWORD, MCP_API_KEY

# 2. Build and start
docker compose up -d

# 3. Run initial import (first time only)
docker exec vault-sync python /app/host/sync.py --vault /vault

# 4. Verify
curl http://localhost:3456/health
```

First startup takes a few minutes — the embeddings service downloads the model (~1.3GB).

## Configuration

All config lives in `.env`. Copy `.env.example` to get started:

| Variable | Required | Default | Description |
|---|---|---|---|
| `VAULT_PATH` | ✅ | — | Absolute path to your notes directory |
| `PGPASSWORD` | ✅ | — | PostgreSQL password |
| `MCP_API_KEY` | ✅ | — | API key for MCP/REST auth (`openssl rand -hex 32`) |
| `PGDATABASE` | | `obsidian` | Database name |
| `PGUSER` | | `postgres` | Database user |
| `SYNC_INTERVAL_SEC` | | `900` | Sync frequency in seconds (15 min) |

## Connecting to Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "memex": {
      "command": "npx",
      "args": [
        "mcp-remote",
        "http://localhost:3456/mcp",
        "--allow-http",
        "--header",
        "X-API-Key: YOUR_MCP_API_KEY"
      ]
    }
  }
}
```

## REST API

All endpoints require `X-API-Key: <key>` header (or `?key=<key>` query param).

```bash
# Health (no auth required)
curl http://localhost:3456/health

# Search
curl -X POST http://localhost:3456/search \
  -H "X-API-Key: $MCP_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"q": "my query", "limit": 5}'

# Stats
curl http://localhost:3456/stats \
  -H "X-API-Key: $MCP_API_KEY"
```

### Search Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `q` | string | required | Search query |
| `limit` | int | 8 | Max results |
| `hyde` | bool | true | Enable HyDE query expansion for short queries |
| `decay` | bool | true | Enable time-decay scoring |
| `decay_boost` | float | 1.0 | Decay intensity (higher = stronger recency preference) |
| `tags` | string[] | — | Filter by frontmatter tags |

## MCP Tools

| Tool | Description |
|---|---|
| `search_vault` | Hybrid RRF search (vector + full-text). Primary tool. |
| `get_note` | Full note content + metadata by path |
| `get_backlinks` | Notes that `[[wikilink]]` to a given note |
| `related_notes` | Semantically similar notes via embeddings |
| `search_by_tag` | Filter notes by frontmatter tags |
| `recent_notes` | Recently modified notes (last N days) |
| `vault_stats` | Note/chunk counts, last sync time, top tags |

## Vault Requirements

Notes should use standard Obsidian frontmatter for best results:

```yaml
---
tags: [reference, tech/networking]
note_type: reference   # reference | project | session | runbook | etc.
---
```

`note_type` and `tags` are indexed and filterable. Everything else is optional — the sync daemon handles plain markdown too.

## Optional: Cloudflare Tunnel (for ChatGPT connector)

To expose the MCP server externally for ChatGPT or other remote clients:

```bash
cloudflared tunnel --url http://localhost:3456
```

Then use `?key=YOUR_API_KEY` query param auth since ChatGPT can't send custom headers.

## Embedding Model

Default model: `mixedbread-ai/mxbai-embed-large-v1` (1024 dimensions, ~560MB).

To use a different model, set `EMBED_MODEL` in the embeddings service environment and update `EMBED_DIM` in the mcp service. The schema will need to be recreated if dimensions change.

## License

MIT
