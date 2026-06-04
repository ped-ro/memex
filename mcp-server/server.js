/**
 * Vault MCP Server
 * Hybrid semantic + full-text search over a personal knowledge base via pgvector.
 *
 * Transports:
 *   - stdio  (default)              → Claude Desktop on workstation
 *   - HTTP   (MCP_TRANSPORT=http)   → OpenClaw, n8n, any HTTP client
 *
 * Tools:
 *   search_vault      — hybrid RRF search (vector + FTS)
 *   get_note          — fetch full note content + metadata
 *   get_backlinks     — notes that link to a given note
 *   related_notes     — semantically similar notes
 *   search_by_tag     — filter by frontmatter tags
 *   recent_notes      — recently modified notes
 *   vault_stats       — counts + last sync time
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import pg from "pg";
import { z } from "zod";
import { readFileSync, existsSync } from "fs";
import { createServer } from "http";
import express from "express";
import crypto from "crypto";
import net from "net";
// ── Instance identity ──────────────────────────────────────────────────────────
const INSTANCE_ID = crypto.randomUUID();
const STARTED_AT = new Date().toISOString();

// ── Config ────────────────────────────────────────────────────────────────────

const DB_CONFIG = {
  host:     process.env.PGHOST     || "localhost",
  port:     parseInt(process.env.PGPORT || "5433"),
  database: process.env.PGDATABASE || "obsidian",
  user:     process.env.PGUSER     || "postgres",
  password: process.env.PGPASSWORD || "",
};

const EMBED_URL = process.env.EMBED_URL || "http://localhost:8765/embed";
const EMBED_DIM = parseInt(process.env.EMBED_DIM || "384");
const HTTP_PORT = parseInt(process.env.MCP_PORT || "3456");
const TRANSPORT = process.env.MCP_TRANSPORT || "stdio";

// ── DB pool ───────────────────────────────────────────────────────────────────

const pool = new pg.Pool({ ...DB_CONFIG, max: 5 });

// Set HNSW ef_search for better recall on vector queries (default 40 is low)
pool.on('connect', (client) => {
  client.query('SET hnsw.ef_search = 80').catch(() => {});
});

async function query(sql, params = []) {
  const client = await pool.connect();
  try {
    return await client.query(sql, params);
  } finally {
    client.release();
  }
}

// ── Embedding ─────────────────────────────────────────────────────────────────

async function fetchWithTimeout(url, options = {}, timeoutMs = 10000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Embed text via local sentence-transformers service.
 * The returned dimension must match the pgvector schema.
 */
async function embedText(text) {
  const resp = await fetchWithTimeout(EMBED_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ texts: [text], is_query: true }),
  }, 30000);
  if (!resp.ok) throw new Error(`Embed service error: ${resp.status}`);
  const data = await resp.json();
  const embedding = data.embeddings?.[0];
  if (!Array.isArray(embedding)) {
    throw new Error("Embed service returned no embedding");
  }
  if (embedding.length !== EMBED_DIM) {
    throw new Error(`Embed dimension mismatch: got ${embedding.length}, expected ${EMBED_DIM}`);
  }
  return embedding;
}

// ── Dedup ─────────────────────────────────────────────────────────────────────

/**
 * Group results by note_path, keep highest-scoring chunk per note.
 * If multiple chunks from the same note matched, merge their context windows
 * by splitting on the separator, deduplicating sections, and rejoining.
 * Returns results sorted by final_score DESC, capped at `limit`.
 */
function deduplicateByNote(results, limit) {
  const byNote = new Map();

  for (const r of results) {
    const key = r.path || r.note_path;
    if (!byNote.has(key)) {
      byNote.set(key, {
        best: r,
        chunks: [{ idx: r.chunkIndex ?? r.chunk_index, context: r.context }],
      });
    } else {
      const entry = byNote.get(key);
      entry.chunks.push({ idx: r.chunkIndex ?? r.chunk_index, context: r.context });
      const bestScore  = parseFloat(entry.best.finalScore  || entry.best.final_score  || entry.best.rrfScore || entry.best.rrf_score || 0);
      const thisScore  = parseFloat(r.finalScore  || r.final_score  || r.rrfScore || r.rrf_score || 0);
      if (thisScore > bestScore) entry.best = r;
    }
  }

  return Array.from(byNote.values())
    .map(({ best, chunks }) => {
      let context = best.context;
      if (chunks.length > 1) {
        // Sort chunks by index, split each context on separator, dedupe sections
        chunks.sort((a, b) => a.idx - b.idx);
        const seen = new Set();
        const sections = [];
        for (const { context: ctx } of chunks) {
          for (const section of ctx.split("\n\n---\n\n")) {
            const trimmed = section.trim();
            if (trimmed && !seen.has(trimmed)) {
              seen.add(trimmed);
              sections.push(trimmed);
            }
          }
        }
        context = sections.join("\n\n---\n\n");
      }
      return { ...best, context, matchedChunks: chunks.length };
    })
    .sort((a, b) => {
      const sa = parseFloat(a.finalScore  || a.final_score  || a.rrfScore || a.rrf_score || 0);
      const sb = parseFloat(b.finalScore  || b.final_score  || b.rrfScore || b.rrf_score || 0);
      return sb - sa;
    })
    .slice(0, limit);
}

// ── Meta-note penalty ─────────────────────────────────────────────────────────
/**
 * Downweight meta/summary notes (session logs, monthly snapshots, continuations,
 * archive) so reference notes dominate technical queries.
 * Applied after RRF scoring, before dedup — so dedup picks the right winner.
 */
const META_PATH_PREFIXES = [
  "030 Reference/Monthly Snapshots/",
  "025 Session Logs/",
  "005 Conversation Continuations/",
  "100 Archive/",
];
const META_PENALTY = 0.7;

function applyMetaPenalty(results) {
  for (const r of results) {
    const p = r.path || r.note_path || "";
    if (META_PATH_PREFIXES.some(prefix => p.startsWith(prefix))) {
      for (const f of ["finalScore", "final_score", "rrfScore", "rrf_score"]) {
        if (r[f] != null) r[f] = String(parseFloat(r[f]) * META_PENALTY);
      }
    }
  }
  // Re-sort by final score after penalty
  results.sort((a, b) => {
    const sa = parseFloat(a.finalScore ?? a.final_score ?? a.rrfScore ?? a.rrf_score ?? 0);
    const sb = parseFloat(b.finalScore ?? b.final_score ?? b.rrfScore ?? b.rrf_score ?? 0);
    return sb - sa;
  });
  return results;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmtNote(row) {
  return {
    path:      row.path,
    title:     row.title,
    tags:      row.tags || [],
    type:      row.note_type,
    date:      row.note_date,
    modified:  row.modified_at,
    wordCount: row.word_count,
  };
}

function readVaultFile(notePath) {
  const vaultRoot = process.env.VAULT_PATH || "/vault";
  try {
    return readFileSync(`${vaultRoot}/${notePath}`, "utf8");
  } catch {
    return null;
  }
}

// ── MCP Server ────────────────────────────────────────────────────────────────

function registerTools(server) {

// ── Tool: search_vault ────────────────────────────────────────────────────────
server.tool(
  "search_vault",
  "Hybrid semantic + keyword search over your personal knowledge base. Combines vector similarity and full-text search using RRF fusion for best results. Use this to find notes by concept, topic, or exact terms.",
  {
    query:        z.string().describe("Search query — natural language or keywords"),
    limit:        z.number().int().min(1).max(20).optional().default(8).describe("Max results to return"),
    tags:         z.array(z.string()).optional().describe("Filter by frontmatter tags (optional)"),
    hyde:            z.boolean().optional().default(true).describe("Use HyDE for better semantic recall (default: true)"),
    decay:           z.boolean().optional().default(true).describe("Apply time-decay scoring (default: true)"),
    decay_boost:     z.number().optional().default(1.0).describe("Decay strength multiplier (1=normal, 2=aggressive, 0=off)"),
    include_context: z.boolean().optional().default(false).describe("Include surrounding chunk context window (default: false to save tokens — the content field has the matched chunk)"),
  },
  async ({ query: q, limit = 8, tags, hyde = true, decay = true, decay_boost = 1.0, include_context = false }) => {
    let embedding;
    try {
      // HyDE-lite: expand very short queries into natural sentences
      let embedQuery = q;
      if (hyde && q.split(/\s+/).length <= 4) {
        embedQuery = `${q} — detailed notes and how-to covering this topic`;
      }
      embedding = await embedText(embedQuery);
    } catch (e) {
      // Fall back to FTS-only if embedding fails
      const fts = await query(`
        SELECT DISTINCT ON (c.note_path)
          c.note_path, n.title, n.tags, n.note_type,
          c.chunk_index, c.content,
          ts_rank_cd(c.content_tsv, plainto_tsquery('english', $1)) AS score
        FROM chunks c
        JOIN notes n ON c.note_path = n.path
        WHERE c.content_tsv @@ plainto_tsquery('english', $1)
          ${tags?.length ? "AND n.tags && $3" : ""}
        ORDER BY c.note_path, score DESC
        LIMIT $2
      `, tags?.length ? [q, limit, tags] : [q, limit]);

      const results = fts.rows.map(r => ({
        path: r.note_path, title: r.title, tags: r.tags,
        type: r.note_type, chunk: r.content, score: r.score,
        searchType: "fts-only (embedding unavailable)",
      }));
      return { content: [{ type: "text", text: JSON.stringify(results, null, 2) }] };
    }

    const embStr = `[${embedding.join(",")}]`;
    const tagFilter = tags?.length ? tags : null;

    // Fetch extra rows so dedup doesn't starve the result set
    const res2 = await query(`
      SELECT * FROM search_hybrid($1, $2::vector, $3, $4)
    `, [q, embStr, limit * 2, tagFilter]);

    // Fetch adjacent chunks for context window expansion
    const paths = res2.rows.map(r => r.note_path);
    const indices = res2.rows.map(r => r.chunk_index);

    const adjRes = await query(`
      SELECT note_path, chunk_index, content
      FROM chunks
      WHERE (note_path, chunk_index) IN (
        SELECT unnest($1::text[]), unnest($2::int[]) - 1
        UNION
        SELECT unnest($1::text[]), unnest($2::int[]) + 1
      )
      ORDER BY note_path, chunk_index
    `, [paths, indices]);

    // Build adjacency map: "path:index" -> content
    const adjMap = {};
    for (const row of adjRes.rows) {
      adjMap[`${row.note_path}:${row.chunk_index}`] = row.content;
    }

    const results = res2.rows.map(r => {
      const prev = adjMap[`${r.note_path}:${r.chunk_index - 1}`] || "";
      const next = adjMap[`${r.note_path}:${r.chunk_index + 1}`] || "";
      const context = [prev, r.content, next].filter(Boolean).join("\n\n---\n\n");
      return {
        path:         r.note_path,
        title:        r.title,
        tags:         r.tags || [],
        type:         r.note_type,
        modified:     r.modified_at,
        chunkIndex:   r.chunk_index,
        content:      r.content,   // matched chunk always present
        context,                   // kept for dedup; stripped below unless requested
        finalScore:   r.final_score,
        rrfScore:     r.rrf_score,
        decayFactor:  r.decay_factor,
        vectorRank:   r.vector_rank,
        ftsRank:      r.fts_rank,
      };
    });

    // Time-decay: boost recent notes in ranking
    if (decay && decay_boost > 0) {
      const now = Date.now();
      for (const r of results) {
        if (r.modified) {
          const ageDays = (now - new Date(r.modified).getTime()) / 86400000;
          // Half-life 90 days, blended 70% original + 30% recency
          const decayFactor = Math.exp(-decay_boost * ageDays / 90);
          const orig = parseFloat(r.finalScore || r.rrfScore || 0);
          r.finalScore = String(orig * (0.7 + 0.3 * decayFactor));
          r.decayFactor = decayFactor.toFixed(4);
        }
      }
    }

    // Meta-note penalty: downweight session logs, snapshots, continuations
    applyMetaPenalty(results);

    const deduped = deduplicateByNote(results, limit);

    // Strip context window unless explicitly requested (saves tokens for MCP clients)
    if (!include_context) {
      for (const r of deduped) delete r.context;
    }

    return { content: [{ type: "text", text: JSON.stringify(deduped, null, 2) }] };
  }
);

// ── Tool: get_note ────────────────────────────────────────────────────────────
server.tool(
  "get_note",
  "Retrieve a note's full content and metadata by its vault path. Use after search_vault to read a specific note in full.",
  {
    path: z.string().describe("Vault-relative path, e.g. '010 Projects/foo.md'"),
  },
  async ({ path: notePath }) => {
    const meta = await query("SELECT * FROM notes WHERE path = $1", [notePath]);
    if (!meta.rows.length) {
      return { content: [{ type: "text", text: `Note not found: ${notePath}` }] };
    }

    const note = meta.rows[0];
    const content = readVaultFile(notePath);

    const result = {
      ...fmtNote(note),
      frontmatter: note.frontmatter,
      wikilinks:   note.wikilinks || [],
      content:     content || "(file not readable — vault may not be mounted)",
    };

    return { content: [{ type: "text", text: JSON.stringify(result, null, 2) }] };
  }
);

// ── Tool: get_backlinks ───────────────────────────────────────────────────────
server.tool(
  "get_backlinks",
  "Find all notes that link to a given note via [[wikilinks]]. Useful for understanding what context surrounds a topic.",
  {
    path: z.string().describe("Vault-relative path of the target note"),
  },
  async ({ path: notePath }) => {
    const res = await query(`
      SELECT source_path, source_title, source_tags
      FROM backlinks
      WHERE note_path = $1
      ORDER BY source_title
    `, [notePath]);

    return {
      content: [{
        type: "text",
        text: JSON.stringify({
          target: notePath,
          backlinks: res.rows.map(r => ({
            path:  r.source_path,
            title: r.source_title,
            tags:  r.source_tags || [],
          })),
          count: res.rows.length,
        }, null, 2),
      }],
    };
  }
);

// ── Tool: related_notes ───────────────────────────────────────────────────────
server.tool(
  "related_notes",
  "Find notes semantically similar to a given note, based on vector embeddings. Good for discovering related topics.",
  {
    path:  z.string().describe("Vault-relative path of the source note"),
    limit: z.number().int().min(1).max(10).optional().default(5),
  },
  async ({ path: notePath, limit = 5 }) => {
    // Get average embedding of the note's chunks as its representation
    const res = await query(`
      SELECT
        n.path,
        n.title,
        n.tags,
        n.note_type,
        1 - (avg_emb <=> src_emb) AS similarity
      FROM (
        SELECT AVG(embedding) AS src_emb
        FROM chunks WHERE note_path = $1
      ) src
      CROSS JOIN LATERAL (
        SELECT
          c.note_path,
          AVG(c.embedding) AS avg_emb
        FROM chunks c
        WHERE c.note_path != $1
        GROUP BY c.note_path
        ORDER BY AVG(c.embedding) <=> src_emb
        LIMIT $2
      ) ranked
      JOIN notes n ON n.path = ranked.note_path
      ORDER BY similarity DESC
    `, [notePath, limit]);

    return {
      content: [{
        type: "text",
        text: JSON.stringify({
          source: notePath,
          related: res.rows.map(r => ({
            path:       r.path,
            title:      r.title,
            tags:       r.tags || [],
            type:       r.note_type,
            similarity: parseFloat(r.similarity).toFixed(4),
          })),
        }, null, 2),
      }],
    };
  }
);

// ── Tool: search_by_tag ───────────────────────────────────────────────────────
server.tool(
  "search_by_tag",
  "Find all notes with specific frontmatter tags. Tags are AND'd together (all must match).",
  {
    tags:  z.array(z.string()).min(1).describe("Tags to filter by"),
    limit: z.number().int().optional().default(20),
  },
  async ({ tags, limit = 20 }) => {
    const res = await query(`
      SELECT path, title, tags, note_type, note_date, modified_at, word_count
      FROM notes
      WHERE tags @> $1
      ORDER BY modified_at DESC
      LIMIT $2
    `, [tags, limit]);

    return {
      content: [{
        type: "text",
        text: JSON.stringify({
          tags,
          notes: res.rows.map(fmtNote),
          count: res.rows.length,
        }, null, 2),
      }],
    };
  }
);

// ── Tool: recent_notes ────────────────────────────────────────────────────────
server.tool(
  "recent_notes",
  "List recently modified notes. Useful for 'what changed lately' or catching up on recent work.",
  {
    days:  z.number().int().min(1).max(365).optional().default(7).describe("How many days back to look"),
    limit: z.number().int().optional().default(15),
  },
  async ({ days = 7, limit = 15 }) => {
    const res = await query(`
      SELECT path, title, tags, note_type, note_date, modified_at, word_count
      FROM notes
      WHERE modified_at >= NOW() - ($1 || ' days')::interval
      ORDER BY modified_at DESC
      LIMIT $2
    `, [days, limit]);

    return {
      content: [{
        type: "text",
        text: JSON.stringify({
          since: `${days} days ago`,
          notes: res.rows.map(fmtNote),
          count: res.rows.length,
        }, null, 2),
      }],
    };
  }
);

// ── Tool: vault_stats ─────────────────────────────────────────────────────────
server.tool(
  "vault_stats",
  "Get vault index statistics: note count, chunk count, last sync time, top tags.",
  {},
  async () => {
    const [counts, lastSync, topTags] = await Promise.all([
      query(`
        SELECT
          (SELECT COUNT(*) FROM notes)  AS notes,
          (SELECT COUNT(*) FROM chunks) AS chunks,
          (SELECT COUNT(*) FROM links)  AS links,
          (SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL) AS embedded
      `),
      query(`SELECT MAX(synced_at) AS last_sync, SUM(notes_added + notes_updated) AS total_synced FROM sync_log`),
      query(`
        SELECT tag, COUNT(*) AS note_count
        FROM notes, UNNEST(tags) AS tag
        GROUP BY tag ORDER BY note_count DESC LIMIT 15
      `),
    ]);

    return {
      content: [{
        type: "text",
        text: JSON.stringify({
          notes:      parseInt(counts.rows[0].notes),
          chunks:     parseInt(counts.rows[0].chunks),
          links:      parseInt(counts.rows[0].links),
          embedded:   parseInt(counts.rows[0].embedded),
          lastSync:   lastSync.rows[0].last_sync,
          topTags:    topTags.rows,
        }, null, 2),
      }],
    };
  }
);



// -- Tool: capture_thought -------------------------------------------------------
server.tool(
  "capture_thought",
  "Capture a quick thought or note into the vault inbox (000 Inbox/ only). Creates a timestamped note with frontmatter. Use for quick capture. Cannot edit or delete existing notes.",
  {
    thought: z.string().describe("The thought or note content to capture"),
    title:   z.string().optional().describe("Short title for the note (auto-generated if omitted)"),
    tags:    z.array(z.string()).optional().describe("Optional tags, e.g. ['#idea', '#work']"),
    source:  z.string().optional().default("chatgpt").describe("Source of the capture (default: chatgpt)"),
  },
  async ({ thought, title, tags, source = "chatgpt" }) => {
    const vaultRoot = process.env.VAULT_PATH || "/vault";
    const inboxDir = `${vaultRoot}/000 Inbox`;

    const now = new Date();
    const pad = (n) => String(n).padStart(2, "0");
    const dateStr = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`;
    const timeStr = `${pad(now.getHours())}${pad(now.getMinutes())}`;
    const slug = title
      ? title.replace(/[^a-zA-Z0-9\s-]/g, "").replace(/\s+/g, " ").trim().slice(0, 60)
      : "Quick capture";
    const filename = `${dateStr} ${timeStr} ${slug}.md`;
    const filepath = `${inboxDir}/${filename}`;

    const fmLines = [
      "---",
      `created: "${dateStr}"`,
      `source: "${source}"`,
      `type: inbox`,
    ];
    if (tags?.length) {
      fmLines.push("tags:");
      for (const tag of tags) {
        const t = tag.startsWith("#") ? tag : `#${tag}`;
        fmLines.push(`  - '${t}'`);
      }
    }
    fmLines.push("---", "");

    const content = fmLines.join("\n") + `# ${title || "Quick Capture"}\n\n${thought}\n`;

    try {
      const { writeFileSync, mkdirSync, existsSync } = await import("fs");
      if (!existsSync(inboxDir)) mkdirSync(inboxDir, { recursive: true });

      if (existsSync(filepath)) {
        return { content: [{ type: "text", text: JSON.stringify({
          error: "File already exists",
          path: `000 Inbox/${filename}`,
        }) }] };
      }

      writeFileSync(filepath, content);
      console.error(`Captured thought: 000 Inbox/${filename} (source: ${source})`);

      return { content: [{ type: "text", text: JSON.stringify({
        ok: true,
        path: `000 Inbox/${filename}`,
        title: title || "Quick Capture",
        source,
        tags: tags || [],
      }, null, 2) }] };
    } catch (e) {
      return { content: [{ type: "text", text: JSON.stringify({ error: e.message }) }] };
    }
  }
);


// ── ChatGPT Compatibility Aliases ─────────────────────────────────────────────
// ChatGPT (non-Developer Mode) requires tools named exactly "search" and "fetch".
// These are thin wrappers around search_vault and get_note respectively.
// All other clients (Claude Desktop, Claude Code, Monina) continue using the
// original tool names — these aliases are additive only.

server.tool(
  "search",
  "Search the knowledge vault for notes matching a query. Returns matching note IDs and summaries. Use this to find relevant information, then use 'fetch' to read full notes.",
  {
    query: z.string().describe("Search query — natural language or keywords"),
    limit: z.number().int().min(1).max(20).optional().default(8).describe("Max results"),
  },
  async ({ query: q, limit = 8 }) => {
    let embedding;
    try {
      let embedQuery = q;
      if (q.split(/\s+/).length <= 4) {
        embedQuery = `${q} — detailed notes and how-to covering this topic`;
      }
      embedding = await embedText(embedQuery);
    } catch (e) {
      const fts = await query(`
        SELECT DISTINCT ON (c.note_path)
          c.note_path, n.title, n.tags, n.note_type, c.content,
          ts_rank_cd(c.content_tsv, plainto_tsquery('english', $1)) AS score
        FROM chunks c JOIN notes n ON c.note_path = n.path
        WHERE c.content_tsv @@ plainto_tsquery('english', $1)
        ORDER BY c.note_path, score DESC LIMIT $2
      `, [q, limit]);
      return { content: [{ type: "text", text: JSON.stringify({
        ids: fts.rows.map(r => r.note_path),
        results: fts.rows.map(r => ({
          id: r.note_path, title: r.title, tags: r.tags || [],
          type: r.note_type, snippet: r.content.slice(0, 300),
        })),
      }, null, 2) }] };
    }

    const embStr = `[${embedding.join(",")}]`;
    const res2 = await query(
      `SELECT * FROM search_hybrid($1, $2::vector, $3, $4)`,
      [q, embStr, limit * 2, null]
    );

    // Time-decay
    const now = Date.now();
    for (const r of res2.rows) {
      if (r.modified_at) {
        const ageDays = (now - new Date(r.modified_at).getTime()) / 86400000;
        const decayFactor = Math.exp(-ageDays / 90);
        const orig = parseFloat(r.rrf_score || 0);
        r.rrf_score = String(orig * (0.7 + 0.3 * decayFactor));
      }
    }

    applyMetaPenalty(res2.rows);
    const deduped = deduplicateByNote(res2.rows, limit);

    return { content: [{ type: "text", text: JSON.stringify({
      ids: deduped.map(r => r.path || r.note_path),
      results: deduped.map(r => ({
        id:    r.path || r.note_path,
        title: r.title,
        tags:  r.tags || [],
        type:  r.type || r.note_type,
        snippet: (r.content || "").slice(0, 300),
      })),
    }, null, 2) }] };
  }
);

server.tool(
  "fetch",
  "Fetch the full content of a note by its ID (vault path). Use after 'search' to read a specific note.",
  {
    id: z.string().describe("Note ID (vault path) from search results, e.g. '020 Domains/Home Lab/foo.md'"),
  },
  async ({ id: notePath }) => {
    const meta = await query("SELECT * FROM notes WHERE path = $1", [notePath]);
    if (!meta.rows.length) {
      return { content: [{ type: "text", text: `Note not found: ${notePath}` }] };
    }

    const note = meta.rows[0];
    const content = readVaultFile(notePath);

    return { content: [{ type: "text", text: JSON.stringify({
      id:          note.path,
      title:       note.title,
      tags:        note.tags || [],
      type:        note.note_type,
      modified:    note.modified_at,
      frontmatter: note.frontmatter,
      content:     content || "(file not readable — vault may not be mounted)",
    }, null, 2) }] };
  }
);

}

// Create server for stdio mode
const server = new McpServer({ name: "memex", version: "1.0.0" });
registerTools(server);

// ── Transport ─────────────────────────────────────────────────────────────────

if (TRANSPORT === "http") {
  const app = express();
  app.use(express.json());
  app.use(express.urlencoded({ extended: false }));

  // Short-lived authorization codes stay in memory; durable clients/tokens live in Postgres.
  const oauthCodes = new Map(); // code -> { client_id, redirect_uri, expires }

  async function oauthTokenValid(token) {
    if (!token) return false;
    const result = await query(`
      SELECT 1
      FROM oauth_tokens
      WHERE token = $1
        AND (expires_at IS NULL OR expires_at > now())
      LIMIT 1
    `, [token]);
    return result.rowCount > 0;
  }

  // ── API Key Auth ──────────────────────────────────────────────────────────
  const API_KEY = process.env.MCP_API_KEY || "";
  if (API_KEY) {
    app.use(async (req, res, next) => {
      try {
        // Public endpoints — no auth required
        const publicPaths = ["/health", "/.well-known/oauth-authorization-server", "/register", "/authorize", "/token"];
        if (publicPaths.includes(req.path)) return next();
        const bearer = (req.headers.authorization || "").replace(/^Bearer\s+/i, "");
        const header  = req.headers["x-api-key"] || "";
        const qparam  = req.query.key || "";
        // API key auth (existing clients)
        if (bearer === API_KEY || header === API_KEY || qparam === API_KEY) return next();
        // OAuth Bearer token auth (ChatGPT)
        if (bearer && await oauthTokenValid(bearer)) return next();
        res.status(401).json({ error: "Unauthorized — invalid or missing API key" });
      } catch (e) {
        res.status(503).json({ error: `Auth check failed: ${e.message}` });
      }
    });
    console.error("🔐 API key auth enabled");
  } else {
    console.error("⚠️  No MCP_API_KEY set — running without auth");
  }



  // ── OAuth 2.0 Endpoints ─────────────────────────────────────────────────

  // Discovery metadata
  app.get("/.well-known/oauth-authorization-server", (req, res) => {
    const issuer = `${req.protocol}://${req.get("host")}`;
    res.json({
      issuer,
      authorization_endpoint: `${issuer}/authorize`,
      token_endpoint: `${issuer}/token`,
      registration_endpoint: `${issuer}/register`,
      response_types_supported: ["code"],
      grant_types_supported: ["authorization_code"],
      token_endpoint_auth_methods_supported: ["client_secret_post", "none"],
      code_challenge_methods_supported: ["S256", "plain"],
    });
  });

  // Dynamic client registration
  app.post("/register", async (req, res) => {
    const clientId = crypto.randomUUID();
    const clientSecret = crypto.randomBytes(32).toString("hex");
    const redirectUris = Array.isArray(req.body.redirect_uris) ? req.body.redirect_uris : [];
    await query(`
      INSERT INTO oauth_clients (client_id, client_secret, redirect_uris)
      VALUES ($1, $2, $3)
    `, [clientId, clientSecret, redirectUris]);
    console.error(`\u{1f511} OAuth client registered: ${clientId}`);
    res.status(201).json({
      client_id: clientId,
      client_secret: clientSecret,
      redirect_uris: redirectUris,
      token_endpoint_auth_method: "client_secret_post",
    });
  });

  // Authorization endpoint — auto-approve (single-user)
  app.get("/authorize", async (req, res) => {
    const { client_id, redirect_uri, state, response_type, code_challenge, code_challenge_method } = req.query;
    if (!client_id || !redirect_uri) {
      return res.status(400).send("Missing client_id or redirect_uri");
    }
    if (response_type && response_type !== "code") {
      return res.status(400).send("Unsupported response_type");
    }
    const clientId = String(client_id);
    const redirectUri = String(redirect_uri);
    const client = await query(`
      SELECT redirect_uris
      FROM oauth_clients
      WHERE client_id = $1
    `, [clientId]);
    if (!client.rowCount) {
      return res.status(400).send("Unknown client_id");
    }
    const registeredRedirects = client.rows[0].redirect_uris || [];
    if (registeredRedirects.length && !registeredRedirects.includes(redirectUri)) {
      return res.status(400).send("redirect_uri is not registered for this client");
    }
    const authCode = crypto.randomBytes(32).toString("hex");
    oauthCodes.set(authCode, {
      client_id: clientId,
      redirect_uri: redirectUri,
      code_challenge: code_challenge || null,
      code_challenge_method: code_challenge_method || null,
      expires: Date.now() + 5 * 60 * 1000,
    });
    console.error(`\u{1f511} OAuth code issued for client: ${clientId}`);
    const url = new URL(redirectUri);
    url.searchParams.set("code", authCode);
    if (state) url.searchParams.set("state", state);
    res.redirect(url.toString());
  });

  // Token exchange
  app.post("/token", async (req, res) => {
    const { grant_type, code, redirect_uri, client_id, client_secret, code_verifier } = req.body;
    if (grant_type !== "authorization_code") {
      return res.status(400).json({ error: "unsupported_grant_type" });
    }
    const codeEntry = oauthCodes.get(code);
    if (!codeEntry) {
      return res.status(400).json({ error: "invalid_grant", error_description: "Invalid or expired authorization code" });
    }
    if (codeEntry.expires < Date.now()) {
      oauthCodes.delete(code);
      return res.status(400).json({ error: "invalid_grant", error_description: "Authorization code expired" });
    }
    if (redirect_uri && redirect_uri !== codeEntry.redirect_uri) {
      return res.status(400).json({ error: "invalid_grant", error_description: "redirect_uri mismatch" });
    }
    if (client_id && client_id !== codeEntry.client_id) {
      return res.status(400).json({ error: "invalid_grant", error_description: "client_id mismatch" });
    }
    const client = await query(`
      SELECT client_secret
      FROM oauth_clients
      WHERE client_id = $1
    `, [codeEntry.client_id]);
    if (!client.rowCount) {
      return res.status(401).json({ error: "invalid_client" });
    }
    const storedSecret = client.rows[0].client_secret;
    if (storedSecret && client_secret && client_secret !== storedSecret) {
      return res.status(401).json({ error: "invalid_client" });
    }
    if (codeEntry.code_challenge) {
      if (!code_verifier) {
        return res.status(400).json({ error: "invalid_grant", error_description: "code_verifier required" });
      }
      let computed;
      if (codeEntry.code_challenge_method === "S256") {
        computed = crypto.createHash("sha256").update(code_verifier).digest("base64url");
      } else {
        computed = code_verifier;
      }
      if (computed !== codeEntry.code_challenge) {
        return res.status(400).json({ error: "invalid_grant", error_description: "PKCE verification failed" });
      }
    }
    oauthCodes.delete(code);
    const accessToken = crypto.randomBytes(48).toString("hex");
    const expiresIn = 86400 * 365;
    const expiresAt = new Date(Date.now() + expiresIn * 1000);
    await query(`
      INSERT INTO oauth_tokens (token, client_id, expires_at)
      VALUES ($1, $2, $3)
    `, [accessToken, codeEntry.client_id, expiresAt]);
    console.error(`\u{1f511} OAuth token issued for client: ${codeEntry.client_id}`);
    res.json({
      access_token: accessToken,
      token_type: "Bearer",
      expires_in: expiresIn,
    });
  });


  // ── Session Management ────────────────────────────────────────────────────
  const sessions = new Map(); // sessionId -> { transport, server }

  function createSessionServer() {
    const s = new McpServer({ name: "memex", version: "1.0.0" });
    registerTools(s);
    return s;
  }

  // MCP over HTTP (Streamable HTTP transport) — session-aware
  app.post("/mcp", async (req, res) => {
    const sessionId = req.headers["mcp-session-id"];

    if (sessionId && sessions.has(sessionId)) {
      const { transport } = sessions.get(sessionId);
      await transport.handleRequest(req, res, req.body);
      return;
    }

    // New session — create transport + server
    // onsessioninitialized fires DURING handleRequest (before it resolves),
    // so the session is stored before the client sends its next request.
    const transport = new StreamableHTTPServerTransport({
      sessionIdGenerator: () => Math.random().toString(36).slice(2),
      onsessioninitialized: (sid) => {
        sessions.set(sid, { transport, server: sessionServer });
        console.error(`\u{1f4e1} New MCP session: ${sid}`);
        transport.onclose = () => {
          sessions.delete(sid);
          console.error(`\u{1f4e1} Session closed: ${sid}`);
        };
      },
    });
    const sessionServer = createSessionServer();
    await sessionServer.connect(transport);
    // Note: handleRequest for SSE may not resolve until stream closes,
    // but onsessioninitialized already stored the session synchronously.
    transport.handleRequest(req, res, req.body).catch(err => {
      console.error("MCP handleRequest error:", err);
    });
  });

  app.get("/mcp", async (req, res) => {
    const sessionId = req.headers["mcp-session-id"];
    if (sessionId && sessions.has(sessionId)) {
      const { transport } = sessions.get(sessionId);
      await transport.handleRequest(req, res);
      return;
    }
    res.status(400).json({ error: "Invalid or missing session ID" });
  });

  app.delete("/mcp", async (req, res) => {
    const sessionId = req.headers["mcp-session-id"];
    if (sessionId && sessions.has(sessionId)) {
      const { transport } = sessions.get(sessionId);
      await transport.close();
      sessions.delete(sessionId);
      console.error(`\u{1f4e1} Session terminated: ${sessionId}`);
    }
    res.status(200).end();
  });

  // Stats endpoint
  app.get("/stats", async (req, res) => {
    try {
      const [counts, lastSync, topTags] = await Promise.all([
        query(`SELECT
          (SELECT COUNT(*) FROM notes)  AS notes,
          (SELECT COUNT(*) FROM chunks) AS chunks,
          (SELECT COUNT(*) FROM links)  AS links,
          (SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL) AS embedded,
          (SELECT COUNT(*)
             FROM notes n
             WHERE NOT EXISTS (SELECT 1 FROM chunks c WHERE c.note_path = n.path)
          ) AS notes_without_chunks`),
        query(`
          SELECT synced_at, status, notes_failed, failed_paths, error, duration_ms
          FROM sync_log
          ORDER BY synced_at DESC
          LIMIT 1
        `),
        query(`SELECT tag, COUNT(*) AS note_count FROM notes, UNNEST(tags) AS tag
               GROUP BY tag ORDER BY note_count DESC LIMIT 10`),
      ]);
      const latestSync = lastSync.rows[0] || {};
      res.json({
        notes:              parseInt(counts.rows[0].notes),
        chunks:             parseInt(counts.rows[0].chunks),
        links:              parseInt(counts.rows[0].links),
        embedded:           parseInt(counts.rows[0].embedded),
        notesWithoutChunks: parseInt(counts.rows[0].notes_without_chunks),
        lastSync:           latestSync.synced_at || null,
        lastSyncStatus:     latestSync.status || null,
        lastSyncFailed:     parseInt(latestSync.notes_failed || 0),
        lastSyncFailedPaths: latestSync.failed_paths || [],
        lastSyncError:      latestSync.error || null,
        topTags:            topTags.rows,
      });
    } catch (e) {
      res.status(500).json({ error: e.message });
    }
  });

  // Review queue endpoint — Claude or Monina can flag a note for human review
  // POST /review { path, note, from }
  // Appends a timestamped entry to 200 Shared/Review Queue.md
  app.post("/review", async (req, res) => {
    const { path: notePath, note, from = "unknown" } = req.body;
    if (!notePath || !note) {
      return res.status(400).json({ error: "path and note are required" });
    }

    const vaultRoot = process.env.VAULT_PATH || "/vault";
    const queuePath = `${vaultRoot}/200 Shared/Review Queue.md`;
    const ts = new Date().toISOString().slice(0, 16).replace("T", " ");
    const noteLink = notePath.replace(/\.md$/, "");

    const entry = `\n## ${ts}\n- **Note:** [[${noteLink}]]\n- **From:** ${from}\n- **Comment:** ${note}\n`;

    try {
      const { readFileSync, writeFileSync, existsSync } = await import("fs");

      let content;
      if (existsSync(queuePath)) {
        content = readFileSync(queuePath, "utf8");
      } else {
        content = [
          "---",
          "title: Review Queue",
          "type: log",
          `tags: [review, monina, claude]`,
          `date: ${ts.slice(0, 10)}`,
          "---",
          "",
          "# Review Queue",
          "",
          "Notes flagged for review by AI agents.",
          "",
        ].join("\n");
      }

      writeFileSync(queuePath, content + entry);
      res.json({ ok: true, path: notePath, queuePath: "200 Shared/Review Queue.md" });
    } catch (e) {
      res.status(500).json({ error: e.message });
    }
  });

  // Health check
  app.get("/health", async (req, res) => {
    try {
      const [dbCheck, syncInfo, counts] = await Promise.all([
        query("SELECT 1"),
        query(`
          SELECT synced_at, status, notes_failed, failed_paths, error
          FROM sync_log
          ORDER BY synced_at DESC
          LIMIT 1
        `),
        query(`SELECT
          (SELECT COUNT(*) FROM notes) AS notes,
          (SELECT COUNT(*) FROM chunks) AS chunks,
          (SELECT COUNT(*)
             FROM notes n
             WHERE NOT EXISTS (SELECT 1 FROM chunks c WHERE c.note_path = n.path)
          ) AS notes_without_chunks`),
      ]);

      const latestSync = syncInfo.rows[0] || {};
      const lastSync = latestSync.synced_at;
      const indexAgeSec = lastSync ? Math.round((Date.now() - new Date(lastSync).getTime()) / 1000) : null;
      const vaultPath = process.env.VAULT_PATH || "/vault";
      const vaultOk = vaultPath ? existsSync(vaultPath) : null;
      const notesWithoutChunks = parseInt(counts.rows[0]?.notes_without_chunks || 0);

      const embedding = {
        health_ok: false,
        probe_ok: false,
        model: null,
        dimension: null,
        error: null,
      };
      try {
        const healthUrl = EMBED_URL.replace(/\/embed$/, "/health");
        const embedHealth = await fetchWithTimeout(healthUrl, {}, 5000);
        if (!embedHealth.ok) throw new Error(`health ${embedHealth.status}`);
        const body = await embedHealth.json();
        embedding.health_ok = true;
        embedding.model = body.model || null;
        embedding.dimension = body.dim || body.dimension || null;
        await embedText("memex health probe");
        embedding.probe_ok = true;
      } catch (e) {
        embedding.error = e.message;
      }

      let status = "ok";
      if (indexAgeSec !== null && indexAgeSec > 3600) status = "critical";
      else if (indexAgeSec !== null && indexAgeSec > 1800) status = "warn";
      if (vaultOk === false) status = "critical";
      if (notesWithoutChunks > 0 && status === "ok") status = "warn";
      if ((latestSync.notes_failed || 0) > 0 && status === "ok") status = "warn";
      if (!embedding.health_ok || !embedding.probe_ok) status = "critical";

      res.json({
        status,
        instance_id: INSTANCE_ID,
        started_at: STARTED_AT,
        container: process.env.HOSTNAME || "unknown",
        db: "connected",
        transport: "http",
        port: HTTP_PORT,
        last_sync_at: lastSync,
        index_age_seconds: indexAgeSec,
        last_sync_status: latestSync.status || null,
        last_sync_failed_notes: parseInt(latestSync.notes_failed || 0),
        last_sync_failed_paths: latestSync.failed_paths || [],
        last_sync_error: latestSync.error || null,
        notes_count: parseInt(counts.rows[0]?.notes || 0),
        chunks_count: parseInt(counts.rows[0]?.chunks || 0),
        notes_without_chunks: notesWithoutChunks,
        embedding,
        vault_mount_ok: vaultOk,
      });
    } catch (e) {
      res.status(503).json({ status: "error", error: e.message });
    }
  });

  // Simple REST search endpoint (for OpenClaw / curl / n8n use without MCP client)
  app.post("/search", async (req, res) => {
    const { q, limit = 8, tags, decay = true, decay_boost = 1.0, include_context = false } = req.body;
    if (!q) return res.status(400).json({ error: "q is required" });
    try {
      // HyDE-lite for REST endpoint
      let embedQuery = q;
      if (q.split(/\s+/).length <= 4) {
        embedQuery = `${q} — detailed notes and how-to covering this topic`;
      }
      let result;
      let searchType = "hybrid";
      try {
        const embedding = await embedText(embedQuery);
        const embStr = `[${embedding.join(",")}]`;
        result = await query(`
          SELECT * FROM search_hybrid($1, $2::vector, $3, $4)
        `, [q, embStr, limit * 2, tags || null]);
      } catch (e) {
        searchType = "fts-only (embedding unavailable)";
        result = await query(`
          SELECT DISTINCT ON (c.note_path)
            c.note_path,
            n.title,
            n.tags,
            n.note_type,
            c.chunk_index,
            c.content,
            n.modified_at,
            ts_rank_cd(c.content_tsv, plainto_tsquery('english', $1)) AS rrf_score
          FROM chunks c
          JOIN notes n ON c.note_path = n.path
          WHERE c.content_tsv @@ plainto_tsquery('english', $1)
            ${tags?.length ? "AND n.tags && $3" : ""}
          ORDER BY c.note_path, rrf_score DESC
          LIMIT $2
        `, tags?.length ? [q, limit * 2, tags] : [q, limit * 2]);
      }

      // Contextual chunks for REST endpoint too
      const paths = result.rows.map(r => r.note_path);
      const idxs  = result.rows.map(r => r.chunk_index);
      let adjMap = {};
      if (paths.length) {
        const adj = await query(`
          SELECT note_path, chunk_index, content FROM chunks
          WHERE (note_path, chunk_index) IN (
            SELECT unnest($1::text[]), unnest($2::int[]) - 1
            UNION
            SELECT unnest($1::text[]), unnest($2::int[]) + 1
          )
        `, [paths, idxs]);
        for (const row of adj.rows) adjMap[`${row.note_path}:${row.chunk_index}`] = row.content;
      }

      const enriched = result.rows.map(r => {
        const prev = adjMap[`${r.note_path}:${r.chunk_index - 1}`] || "";
        const next = adjMap[`${r.note_path}:${r.chunk_index + 1}`] || "";
        return { ...r, context: [prev, r.content, next].filter(Boolean).join("\n\n---\n\n") };
      });

      // Time-decay for REST endpoint
      if (decay && decay_boost > 0) {
        const now = Date.now();
        for (const r of enriched) {
          if (r.modified_at) {
            const ageDays = (now - new Date(r.modified_at).getTime()) / 86400000;
            const decayFactor = Math.exp(-decay_boost * ageDays / 90);
            const orig = parseFloat(r.rrf_score || 0);
            r.rrf_score = String(orig * (0.7 + 0.3 * decayFactor));
            r.decay_factor = decayFactor.toFixed(4);
          }
        }
      }

      // Meta-note penalty + dedup
      applyMetaPenalty(enriched);
      const deduped = deduplicateByNote(enriched, limit);

      // Strip context unless explicitly requested
      if (!include_context) {
        for (const r of deduped) delete r.context;
      }

      res.json({ query: q, searchType, results: deduped });
    } catch (e) {
      res.status(500).json({ error: e.message });
    }
  });

  // ── Port-in-use guard ──────────────────────────────────────────────────────
  const portCheck = net.createServer();
  portCheck.once("error", (err) => {
    if (err.code === "EADDRINUSE") {
      console.error(`FATAL: Port ${HTTP_PORT} already in use — aborting startup`);
      process.exit(1);
    }
  });
  portCheck.once("listening", () => {
    portCheck.close(() => {
      app.listen(HTTP_PORT, "0.0.0.0", () => {
        console.error(`✅ Vault MCP server (HTTP) listening on :${HTTP_PORT}  [instance: ${INSTANCE_ID}]`);
        console.error(`   MCP endpoint: http://0.0.0.0:${HTTP_PORT}/mcp`);
        console.error(`   Health:       http://0.0.0.0:${HTTP_PORT}/health`);
        console.error(`   REST search:  http://0.0.0.0:${HTTP_PORT}/search`);
      });
    });
  });
  portCheck.listen(HTTP_PORT, "0.0.0.0");
} else {
  // Default: stdio transport
  const transport = new StdioServerTransport();
  await server.connect(transport);
  process.stderr.write("Vault MCP server running (stdio)\n");
}
