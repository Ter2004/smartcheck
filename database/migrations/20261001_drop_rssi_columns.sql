-- Drop the RSSI columns. No RSSI check has run at check-in since 027318c, and
-- 43f5f7b removed the last code that read or wrote them; room presence is the
-- board's signed nonce (BLE) or the TOTP code.
-- Checked before dropping (1 Oct 2026): no attendance row had a ble_rssi value,
-- the only beacon had the default rssi_threshold -75, and no function, view or
-- policy referenced either column.
-- Apply only after every running server is on 43f5f7b or later.
ALTER TABLE public.attendance DROP COLUMN IF EXISTS ble_rssi;
ALTER TABLE public.beacons    DROP COLUMN IF EXISTS rssi_threshold;
