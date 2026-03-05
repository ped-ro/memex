-- Vault Search Schema
-- pgvector 0.8.2 (Docker local)
-- Hybrid search: vector (HNSW) + full-text (tsvector) + wiki-link graph
-- Embeddings: sentence-transformers all-MiniLM-L6-v2 (384 dimensions)

-- ============================================================
-- NOTES: one row per vault file, metadata only
-- ============================================================
CREATE TABLE IF NOT EXISTS notes (
    path            TEXT PRIMARY KEY,           -- relative vault path, e.g. "010 Projects/foo.md"
    title           TEXT NOT NULL,              -- note title (H1 or filename)
    tags            TEXT[],                     -- frontmatter tags array
    note_type       TEXT,                       -- frontmatter "type" field (runbook, log, reference, etc.)
    note_date       DATE,                       -- frontmatter "date" field
    created_at      TIMESTAMPTZ,                -- file ctime
    modified_at     TIMESTAMPTZ,                -- file mtime (used for incremental sync)
    frontmatter     JSONB,                      -- full frontmatter as JSONB (flexible, queryable)
    wikilinks       TEXT[],                     -- outgoing [[wikilinks]] extracted from content
    word_count      INT,                        -- rough word count for context budget planning
    indexed_at      TIMESTAMPTZ DEFAULT now()   -- when we last indexed this note
);

CREATE INDEX IF NOT EXISTS notes_tags_gin    ON notes USING GIN (tags);
CREATE INDEX IF NOT EXISTS notes_type_idx    ON notes (note_type);
CREATE INDEX IF NOT EXISTS notes_modified_idx ON notes (modified_at DESC);
CREATE INDEX IF NOT EXISTS notes_frontmatter_gin ON notes USING GIN (frontmatter);

-- ============================================================
-- CHUNKS: text chunks with embeddings + full-text search
-- ============================================================
CREATE TABLE IF NOT EXISTS chunks (
    id              BIGSERIAL PRIMARY KEY,
    note_path       TEXT NOT NULL REFERENCES notes(path) ON DELETE CASCADE,
    chunk_index     INT NOT NULL,               -- position within note (0-based)
    content         TEXT NOT NULL,              -- raw chunk text
    token_count     INT,                        -- approximate token count
    embedding       vector(384),                -- sentence-transformers all-MiniLM-L6-v2
    content_tsv     TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    UNIQUE (note_path, chunk_index)
);

-- HNSW index for fast approximate nearest-neighbor search
-- m=16, ef_construction=64 — good defaults for <100k chunks
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks 
    USING hnsw (embedding vector_cosine_ops) 
    WITH (m = 16, ef_construction = 64);

-- GIN index for full-text search
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING GIN (content_tsv);

-- For fast lookup of all chunks belonging to a note
CREATE INDEX IF NOT EXISTS chunks_note_path_idx ON chunks (note_path, chunk_index);

-- ============================================================
-- LINKS: directed wiki-link graph
-- ============================================================
CREATE TABLE IF NOT EXISTS links (
    source_path     TEXT NOT NULL REFERENCES notes(path) ON DELETE CASCADE,
    target_path     TEXT,                       -- may be NULL if target note doesn't exist yet
    target_title    TEXT NOT NULL,              -- raw wikilink text (unresolved)
    PRIMARY KEY (source_path, target_title)
);

CREATE INDEX IF NOT EXISTS links_target_path_idx ON links (target_path);
CREATE INDEX IF NOT EXISTS links_source_path_idx ON links (source_path);

-- ============================================================
-- HYBRID SEARCH FUNCTION
-- Combines vector similarity + full-text via Reciprocal Rank Fusion (RRF)
-- ============================================================
CREATE OR REPLACE FUNCTION rrf_score(rank BIGINT, rrf_k INT DEFAULT 50)
RETURNS NUMERIC
LANGUAGE SQL IMMUTABLE PARALLEL SAFE AS $$
    SELECT COALESCE(1.0 / ($1 + $2), 0.0);
$$;

CREATE OR REPLACE FUNCTION search_hybrid(
    query_text      TEXT,
    query_embedding vector,
    result_limit    INT DEFAULT 10,
    tag_filter      TEXT[] DEFAULT NULL
)
RETURNS TABLE (
    note_path       TEXT,
    title           TEXT,
    tags            TEXT[],
    note_type       TEXT,
    chunk_index     INT,
    content         TEXT,
    rrf_score       NUMERIC,
    vector_rank     BIGINT,
    fts_rank        BIGINT,
    modified_at     TIMESTAMPTZ
)
LANGUAGE SQL STABLE AS $$
    WITH
    -- vector similarity search
    vec AS (
        SELECT
            c.note_path,
            c.chunk_index,
            c.content,
            ROW_NUMBER() OVER (ORDER BY c.embedding <=> query_embedding) AS rank
        FROM chunks c
        JOIN notes n ON c.note_path = n.path
        WHERE tag_filter IS NULL OR n.tags && tag_filter
        ORDER BY c.embedding <=> query_embedding
        LIMIT result_limit * 3
    ),
    -- full-text search
    fts AS (
        SELECT
            c.note_path,
            c.chunk_index,
            c.content,
            ROW_NUMBER() OVER (ORDER BY ts_rank_cd(c.content_tsv, query) DESC) AS rank
        FROM chunks c
        JOIN notes n ON c.note_path = n.path,
             to_tsquery('english', 
                 regexp_replace(
                     regexp_replace(trim(query_text), '\s+', ' & ', 'g'),
                     '[^a-zA-Z0-9 &]', '', 'g'
                 )
             ) query
        WHERE c.content_tsv @@ query
          AND (tag_filter IS NULL OR n.tags && tag_filter)
        ORDER BY ts_rank_cd(c.content_tsv, query) DESC
        LIMIT result_limit * 3
    ),
    -- RRF fusion
    fused AS (
        SELECT
            COALESCE(v.note_path, f.note_path) AS note_path,
            COALESCE(v.chunk_index, f.chunk_index) AS chunk_index,
            COALESCE(v.content, f.content) AS content,
            rrf_score(COALESCE(v.rank, 999999)) + rrf_score(COALESCE(f.rank, 999999)) AS score,
            v.rank AS vector_rank,
            f.rank AS fts_rank
        FROM vec v
        FULL OUTER JOIN fts f ON v.note_path = f.note_path AND v.chunk_index = f.chunk_index
    )
    SELECT
        fused.note_path,
        n.title,
        n.tags,
        n.note_type,
        fused.chunk_index,
        fused.content,
        fused.score,
        fused.vector_rank,
        fused.fts_rank,
        n.modified_at
    FROM fused
    JOIN notes n ON fused.note_path = n.path
    ORDER BY fused.score DESC
    LIMIT result_limit;
$$;

-- ============================================================
-- BACKLINKS VIEW: inbound links to a note
-- ============================================================
CREATE OR REPLACE VIEW backlinks AS
    SELECT
        l.target_path AS note_path,
        l.source_path,
        n.title AS source_title,
        n.tags AS source_tags
    FROM links l
    JOIN notes n ON l.source_path = n.path
    WHERE l.target_path IS NOT NULL;

-- ============================================================
-- SYNC TRACKING: for incremental updates
-- ============================================================
CREATE TABLE IF NOT EXISTS sync_log (
    id              BIGSERIAL PRIMARY KEY,
    synced_at       TIMESTAMPTZ DEFAULT now(),
    notes_added     INT DEFAULT 0,
    notes_updated   INT DEFAULT 0,
    notes_deleted   INT DEFAULT 0,
    chunks_total    INT DEFAULT 0,
    duration_ms     INT
);
