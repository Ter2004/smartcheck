-- Broadcast room board (firmware/smartcheck_broadcast, app/services/broadcast_beacon.py)
-- Needed only for CHECKIN_PROXIMITY_METHOD=broadcast. Safe to re-run.
--
-- broadcast_sightings   first time each signed counter was seen per board; a counter
--                       first seen longer ago than 30 s, or far behind the newest one,
--                       is a replay. Server-only (service key); no client access.

CREATE TABLE IF NOT EXISTS public.broadcast_sightings (
    beacon_id      uuid        NOT NULL REFERENCES public.beacons (id) ON DELETE CASCADE,
    counter        bigint      NOT NULL,
    first_seen_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (beacon_id, counter)
);

ALTER TABLE public.broadcast_sightings ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.broadcast_sightings FROM PUBLIC, anon, authenticated;

-- ── Verification (the last row must read END) ───────────────────────────────
-- SELECT 'table', count(*)::text FROM information_schema.tables WHERE table_name = 'broadcast_sightings'
-- UNION ALL SELECT 'rls', relrowsecurity::text FROM pg_class WHERE relname = 'broadcast_sightings'
-- UNION ALL SELECT 'END', 'END';
