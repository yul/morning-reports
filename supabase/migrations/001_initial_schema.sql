-- ============================================================
-- morning-reports: initial schema
-- Run this in Supabase SQL editor (or supabase/migrations/)
-- ============================================================

-- Enable pgvector extension
create extension if not exists vector;


-- ============================================================
-- filings
-- Lightweight metadata for each downloaded 10-K / 10-Q.
-- One row per accession number.
-- ============================================================
create table if not exists filings (
    id                  bigint generated always as identity primary key,
    accession_number    text        not null unique,     -- "0000320193-24-000001"
    ticker              text        not null,
    company_name        text        not null,
    cik                 integer     not null,
    form                text        not null,            -- "10-K" | "10-Q"
    filing_date         date        not null,
    period_of_report    date,
    url                 text,
    ingested_at         timestamptz not null default now(),

    -- JSON blob of flat KPI metrics from XBRL (revenue, net_income, …)
    -- Nullable: older filings may lack XBRL data
    financial_metrics   jsonb
);

create index if not exists filings_ticker_idx       on filings (ticker);
create index if not exists filings_filing_date_idx  on filings (filing_date desc);
create index if not exists filings_form_idx         on filings (form);


-- ============================================================
-- chunks
-- One row per embeddable text unit extracted from a filing.
-- ============================================================
create table if not exists chunks (
    id                  bigint generated always as identity primary key,
    filing_id           bigint      not null references filings (id) on delete cascade,
    accession_number    text        not null,            -- denormalised for fast lookup

    -- Section identity
    item_key            text        not null,            -- "Item 7" | "financials:income_statement"
    section_title       text        not null,            -- "MD&A" | "Income Statement"
    is_financial        boolean     not null default false,

    -- Position within section
    chunk_index         integer     not null,
    total_chunks        integer     not null,

    -- Content
    text                text        not null,
    char_count          integer     not null,

    -- Embedding (voyage-3-lite = 512 dims)
    embedding           vector(512),

    embedded_at         timestamptz,
    created_at          timestamptz not null default now()
);

create index if not exists chunks_filing_id_idx         on chunks (filing_id);
create index if not exists chunks_accession_idx         on chunks (accession_number);
create index if not exists chunks_item_key_idx          on chunks (item_key);
create index if not exists chunks_is_financial_idx      on chunks (is_financial);

-- HNSW index for fast approximate nearest-neighbour search.
-- cosine distance works well for normalised Voyage AI embeddings.
-- ef_construction=128, m=16 are sensible defaults for up to ~1M vectors.
create index if not exists chunks_embedding_idx
    on chunks using hnsw (embedding vector_cosine_ops)
    with (m = 16, ef_construction = 128);


-- ============================================================
-- scores
-- Cached investment scores for a filing under a given scoring rule.
-- Invalidated when the rule_hash changes.
-- ============================================================
create table if not exists scores (
    id              bigint generated always as identity primary key,
    filing_id       bigint      not null references filings (id) on delete cascade,
    rule_hash       text        not null,   -- sha256 of the scoring prompt text
    score           numeric(6,3) not null,  -- e.g. 7.250
    rationale       text,                   -- LLM explanation
    scored_at       timestamptz not null default now(),

    unique (filing_id, rule_hash)
);

create index if not exists scores_filing_id_idx on scores (filing_id);
create index if not exists scores_rule_hash_idx on scores (rule_hash);


-- ============================================================
-- prices
-- Daily OHLCV cache for backtesting.
-- ============================================================
create table if not exists prices (
    id          bigint generated always as identity primary key,
    ticker      text    not null,
    date        date    not null,
    open        numeric(12,4),
    high        numeric(12,4),
    low         numeric(12,4),
    close       numeric(12,4),
    volume      bigint,

    unique (ticker, date)
);

create index if not exists prices_ticker_date_idx on prices (ticker, date desc);


-- ============================================================
-- backtest_runs
-- One row per user-initiated backtest.
-- ============================================================
create table if not exists backtest_runs (
    id              bigint generated always as identity primary key,
    rule_hash       text        not null,
    allocation_rule text        not null,   -- user's free-form allocation description
    score_rule      text        not null,   -- user's free-form scoring description
    start_date      date        not null,
    end_date        date        not null,
    use_close       boolean     not null default false,  -- false = trade on open
    result          jsonb,                  -- {equity_curve, trades, metrics}
    status          text        not null default 'pending',  -- pending|running|done|error
    error           text,
    created_at      timestamptz not null default now(),
    completed_at    timestamptz
);

create index if not exists backtest_runs_rule_hash_idx  on backtest_runs (rule_hash);
create index if not exists backtest_runs_status_idx     on backtest_runs (status);


-- ============================================================
-- match_chunks (RPC helper)
-- Similarity search: returns top-k chunks closest to a query vector.
-- Called from Python as: client.rpc("match_chunks", {...})
-- ============================================================
create or replace function match_chunks(
    query_embedding     vector(512),
    match_count         int         default 10,
    filter_ticker       text        default null,
    filter_form         text        default null,
    filter_item_key     text        default null,
    min_date            date        default null,
    max_date            date        default null
)
returns table (
    chunk_id            bigint,
    filing_id           bigint,
    accession_number    text,
    ticker              text,
    form                text,
    filing_date         date,
    period_of_report    date,
    item_key            text,
    section_title       text,
    is_financial        boolean,
    chunk_index         integer,
    total_chunks        integer,
    text                text,
    similarity          float
)
language plpgsql
as $$
begin
    return query
    select
        c.id              as chunk_id,
        c.filing_id,
        c.accession_number,
        f.ticker,
        f.form,
        f.filing_date,
        f.period_of_report,
        c.item_key,
        c.section_title,
        c.is_financial,
        c.chunk_index,
        c.total_chunks,
        c.text,
        1 - (c.embedding <=> query_embedding) as similarity
    from chunks c
    join filings f on f.id = c.filing_id
    where
        c.embedding is not null
        and (filter_ticker   is null or f.ticker      = filter_ticker)
        and (filter_form     is null or f.form         = filter_form)
        and (filter_item_key is null or c.item_key     = filter_item_key)
        and (min_date        is null or f.filing_date >= min_date)
        and (max_date        is null or f.filing_date <= max_date)
    order by c.embedding <=> query_embedding
    limit match_count;
end;
$$;
