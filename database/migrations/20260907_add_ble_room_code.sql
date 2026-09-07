-- Adds the room-identifier value the ESP32 BLE peripheral exposes over GATT
-- (see firmware/README.md), so /api/checkin can verify it against the
-- session's beacon when CHECKIN_PROXIMITY_METHOD=ble. NULL means "not
-- configured for BLE check-in" -- treated as a system failure (503), not a
-- wrong-room rejection.
ALTER TABLE beacons ADD COLUMN IF NOT EXISTS ble_room_code text;
COMMENT ON COLUMN beacons.ble_room_code IS
    'Exact room-identifier string read from the ESP32 BLE GATT characteristic (e.g. TEST-101). NULL = BLE check-in not configured for this beacon.';

-- Test beacon: must match firmware/smartcheck_ble/smartcheck_ble.ino's
-- ROOM_ID constant exactly ("TEST-101", 8 ASCII bytes) or every BLE
-- check-in against this beacon hits the 503 misconfiguration path.
UPDATE beacons SET ble_room_code = 'TEST-101'
WHERE id = 'aa000000-0000-0000-0000-000000000001';
