-- LINE Beacon proximity (optional, LINE_BEACON_ENABLED=1). The LINE app detects
-- the room board (LINE Simple Beacon) and LINE's server posts a webhook to
-- /line/webhook; the server then issues the same proximity receipt the BLE
-- button flow issues. Nothing here changes existing rows.
--
-- users.line_user_id      LINE userId linked once by the student (link code)
-- beacons.line_hwid       5-byte LINE Simple Beacon HWID (10 hex chars) of the room board
-- line_link_codes         one pending link code per student (hash only, 10 minutes)
-- line_beacon_sightings   first time each signed device-message counter was seen per
--                         board; a counter older than the freshness window is a replay

ALTER TABLE public.users   ADD COLUMN IF NOT EXISTS line_user_id text;
ALTER TABLE public.beacons ADD COLUMN IF NOT EXISTS line_hwid text;
CREATE UNIQUE INDEX IF NOT EXISTS users_line_user_id_key ON public.users (line_user_id) WHERE line_user_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS beacons_line_hwid_key ON public.beacons (line_hwid) WHERE line_hwid IS NOT NULL;
ALTER TABLE public.beacons DROP CONSTRAINT IF EXISTS beacons_line_hwid_format;
ALTER TABLE public.beacons ADD CONSTRAINT beacons_line_hwid_format
    CHECK (line_hwid IS NULL OR line_hwid ~ '^[0-9a-f]{10}$');

CREATE TABLE IF NOT EXISTS public.line_link_codes (
    user_id     uuid        PRIMARY KEY REFERENCES public.users (id) ON DELETE CASCADE,
    code_hash   text        NOT NULL UNIQUE,
    expires_at  timestamptz NOT NULL
);

CREATE TABLE IF NOT EXISTS public.line_beacon_sightings (
    beacon_id      uuid        NOT NULL REFERENCES public.beacons (id) ON DELETE CASCADE,
    counter        bigint      NOT NULL,
    first_seen_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (beacon_id, counter)
);

ALTER TABLE public.line_link_codes       ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.line_beacon_sightings ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.line_link_codes       FROM PUBLIC, anon, authenticated;
REVOKE ALL ON public.line_beacon_sightings FROM PUBLIC, anon, authenticated;

-- ── Verification (the last row must read END) ───────────────────────────────
-- SELECT 'users.line_user_id' AS item, count(*)::text AS value FROM information_schema.columns WHERE table_name='users' AND column_name='line_user_id'
-- UNION ALL SELECT 'beacons.line_hwid', count(*)::text FROM information_schema.columns WHERE table_name='beacons' AND column_name='line_hwid'
-- UNION ALL SELECT 'rls_' || relname, relrowsecurity::text FROM pg_class WHERE relname IN ('line_link_codes', 'line_beacon_sightings')
-- UNION ALL SELECT 'END', 'END';
