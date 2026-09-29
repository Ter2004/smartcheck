-- Per-room HMAC key for BLE challenge-response check-in (server-only column;
-- anon/authenticated have no table access since the 2026-09-29 lockdown).
ALTER TABLE public.beacons ADD COLUMN IF NOT EXISTS ble_secret text;
ALTER TABLE public.beacons ADD CONSTRAINT beacons_ble_secret_format
    CHECK (ble_secret IS NULL OR ble_secret ~ '^[0-9a-f]{64}$');
COMMENT ON COLUMN public.beacons.ble_secret IS
    'Hex 32-byte HMAC-SHA256 key shared with the room ESP32 (firmware secret.h). NULL = BLE challenge not configured: check-in answers 503.';
NOTIFY pgrst, 'reload schema';
