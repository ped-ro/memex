# memex

A personal knowledge engine. Semantic search, MCP server, and sync stack for your second brain — exposed to Claude, ChatGPT, and any MCP/HTTP client.

**No cloud dependency required. Everything can run on your machine.**

## What It Does

- Hybrid search combining vector similarity + full-text (RRF fusion) over your vault
- HyDE query expansion for short/conversational queries
- Time-decay scoring to surface recent notes
- Meta-note penalty to keep reference docs above session logs
- 10 MCP tools for search, retrieval, graph queries, and quick capture
- ChatGPT-compatible `search`/`fetch` aliases (no Developer Mode required)
- REST endpoint for non-MCP clients (n8n, scripts, curl)
- Incremental sync every 15 minutes — only re-embeds changed notes
- API key auth (Bearer, X-API-Key header, or query param)

## Architecture

```
Obsidian Vault (local)
    ↓ (mount)
sync daemon  →  embeddings service (local or OpenAI)
    ↓
pgvector (PostgreSQL 17 + pgvector extension)
    ↓
MCP server (Node.js, port 3456)
    ├── Claude Desktop / Claude Code (MCP transport)
    ├── ChatGPT (MCP connector — no Developer Mode needed)
    └── Any HTTP client (REST API)
```

## Requirements

- Docker + Docker Compose
- Your notes accessible as a local directory (Obsidian, Markdown, etc.)

**Local embeddings (default):** ~4GB disk for model download, ~500MB RAM at idle
**OpenAI embeddings (optional):** OpenAI API key, minimal local resources, ~$0.01 per full index

## Quick Start

```bash
git clone https://github.com/ped-ro/memex
cd memex

# 1. Configure
cp .env.example .env
# Edit .env — set VAULT_PATH, PGPASSWORD, MCP_API_KEY

# 2. Build and start
docker compose up -d

# 3. Run initial import (first time only — takes ~38 min with local model on M1)
docker exec -e PYTHONUNBUFFERED=1 vault-sync python /app/host/sync.py --vault /vault

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
| `OPENAI_API_KEY` | | — | Only needed if using OpenAI embeddings |
| `PGDATABASE` | | `obsidian` | Database name |
| `PGUSER` | | `postgres` | Database user |
| `SYNC_INTERVAL_SEC` | | `900` | Sync frequency in seconds (15 min) |

## Using OpenAI Embeddings (Optional)

The default stack uses a local embedding model (`mxbai-embed-large-v1`, 1024-dim) — no API calls, no cost. If you prefer OpenAI's `text-embedding-3-small` (1536-dim, faster startup, smaller footprint):

```bash
# Add your key to .env
echo "OPENAI_API_KEY=sk-..." >> .env

# Start with the override
docker compose -f docker-compose.yml -f docker-compose.openai.yml up -d
```

**Switching from local to OpenAI** requires a full re-index (different vector dimensions). Drop the chunks table and re-import:

```bash
docker exec vault-pgvector psql -U postgres -d obsidian -c "
  DROP INDEX IF EXISTS chunks_embedding_hnsw;
  ALTER TABLE chunks ALTER COLUMN embedding TYPE vector(1536) USING NULL::vector(1536);
  CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=64);
  TRUNCATE chunks;
"
docker exec -e PYTHONUNBUFFERED=1 vault-sync python /app/host/sync.py --vault /vault
```

## Connecting Clients

### Claude Desktop

Add to `claude_desktop_config.json` (quit Claude Desktop first — it overwrites config on shutdown):

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

### Claude Code

```bash
claude mcp add --transport http memex http://localhost:3456/mcp \
  -H "X-API-Key: YOUR_MCP_API_KEY" --scope user
```

### ChatGPT (Plus or Pro)

**No Developer Mode needed.** The server exposes `search` and `fetch` alias tools that satisfy ChatGPT's non-Developer Mode connector requirements — your ChatGPT memory stays enabled.

ChatGPT can't send custom headers, so use query param auth:

1. Settings → Apps & Connectors → **Add new connector**
2. **URL:** `https://your-domain.com/mcp?key=YOUR_MCP_API_KEY`
3. **Auth:** No Authentication (key is in the URL)

Requires a public endpoint — use [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/) or similar.

### Other MCP Clients (Gemini, etc.)

Any client supporting MCP Streamable HTTP transport can connect at `http://localhost:3456/mcp` with API key auth.

## REST API

All endpoints except `/health` require auth (`X-API-Key` header, `Authorization: Bearer`, or `?key=` query param).

```bash
# Health (no auth required)
curl http://localhost:3456/health

# Search
curl -X POST http://localhost:3456/search \
  -H "X-API-Key: $MCP_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"q": "my query", "limit": 5}'

# Stats
curl http://localhost:3456/stats -H "X-API-Key: $MCP_API_KEY"
```

### Search Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `q` | string | required | Search query |
| `limit` | int | 8 | Max results |
| `hyde` | bool | true | HyDE query expansion for short queries |
| `decay` | bool | true | Time-decay scoring |
| `decay_boost` | float | 1.0 | Decay intensity (higher = stronger recency bias) |
| `tags` | string[] | — | Filter by frontmatter tags |

## MCP Tools

| Tool | Type | Description |
|---|---|---|
| `search_vault` | read | Hybrid RRF search (vector + full-text). Primary search tool. |
| `get_note` | read | Full note content + metadata by vault path |
| `get_backlinks` | read | Notes that `[[wikilink]]` to a given note |
| `related_notes` | read | Semantically similar notes via embeddings |
| `search_by_tag` | read | Filter notes by frontmatter tags |
| `recent_notes` | read | Recently modified notes (last N days) |
| `vault_stats` | read | Note/chunk counts, last sync time, top tags |
| `capture_thought` | write | Quick capture to inbox (`000 Inbox/` only) |
| `search` | read | ChatGPT alias for `search_vault` — returns `{ids, results}` |
| `fetch` | read | ChatGPT alias for `get_note` — takes `id` instead of `path` |

The `search`/`fetch` aliases exist so ChatGPT can connect without Developer Mode (which disables ChatGPT's memory). All other clients should use `search_vault`/`get_note`.

## Vault Requirements

Notes should use standard Obsidian frontmatter for best results:

```yaml
---
tags: [reference, tech/networking]
note_type: reference   # reference | project | session | runbook | etc.
---
```

`note_type` and `tags` are indexed and filterable. Plain markdown without frontmatter works too — the sync daemon handles it.

## Embedding Models

**Local (default):** `mixedbread-ai/mxbai-embed-large-v1` — 1024 dimensions, ~560MB download, runs entirely on your machine.

**OpenAI (optional):** `text-embedding-3-small` — 1536 dimensions, requires API key, ~$0.01 for a 200-note vault. Use `docker-compose.openai.yml` override.

To switch models after initial indexing, you must recreate the HNSW index with the new dimension and re-embed everything. See "Using OpenAI Embeddings" above.

## License

MIT
