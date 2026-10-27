"""Broadcast room board: payload vector shared with the firmware, freshness, proximity route."""
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

from flask import Flask

import app
from app.routes import api_checkin as route
from app.services import broadcast_beacon, proximity_receipt

KEY = bytes(range(32)).hex()
# Same bytes as selfTest() in firmware/smartcheck_broadcast/smartcheck_broadcast.ino.
FIRMWARE_VECTOR = "534301544553542d313031010203047b6919855429e731"


class PayloadTests(unittest.TestCase):
    def test_firmware_vector(self):
        self.assertEqual(broadcast_beacon.payload(KEY, "TEST-101", 0x01020304).hex(), FIRMWARE_VECTOR)
        self.assertEqual(broadcast_beacon.verify(KEY, "TEST-101", FIRMWARE_VECTOR), 0x01020304)

    def test_upper_case_hex_and_short_room_code(self):
        self.assertEqual(broadcast_beacon.verify(KEY, "TEST-101", FIRMWARE_VECTOR.upper()), 0x01020304)
        short = broadcast_beacon.payload(KEY, "R1", 9).hex()
        self.assertEqual(broadcast_beacon.verify(KEY, "R1", short), 9)

    def test_refused(self):
        tampered = bytearray(bytes.fromhex(FIRMWARE_VECTOR))
        tampered[-1] ^= 1
        other_room = broadcast_beacon.payload(KEY, "ROOM-202", 5).hex()
        for value in [tampered.hex(), FIRMWARE_VECTOR[:-2], "zz" * 23, None, 7,
                      "5344" + FIRMWARE_VECTOR[4:]]:
            self.assertIsNone(broadcast_beacon.verify(KEY, "TEST-101", value), value)
        self.assertIsNone(broadcast_beacon.verify("ff" * 32, "TEST-101", FIRMWARE_VECTOR))
        self.assertIsNone(broadcast_beacon.verify(None, "TEST-101", FIRMWARE_VECTOR))
        self.assertIsNone(broadcast_beacon.verify(KEY, "TEST-101", other_room))   # another room's board
        self.assertIsNone(broadcast_beacon.verify(KEY, None, FIRMWARE_VECTOR))


class Sightings:
    """In-memory broadcast_sightings with the query chains fresh() uses."""
    def __init__(self):
        self.rows = {}

    def table(self, name):
        assert name == "broadcast_sightings"
        return Query(self)


class Query:
    def __init__(self, db):
        self.db, self.filters, self.newest, self.write = db, {}, False, None

    def select(self, *_):
        return self

    def eq(self, column, value):
        self.filters[column] = value
        return self

    def order(self, *_, **__):
        self.newest = True
        return self

    def limit(self, *_):
        return self

    def maybe_single(self):
        return self

    def upsert(self, row, **_):
        self.write = row
        return self

    def execute(self):
        if self.write:
            self.db.rows.setdefault((self.write["beacon_id"], self.write["counter"]), self.write["first_seen_at"])
            return SimpleNamespace(data=None)
        beacon = self.filters["beacon_id"]
        if self.newest:
            counters = sorted((c for b, c in self.db.rows if b == beacon), reverse=True)
            return SimpleNamespace(data=[{"counter": c} for c in counters[:1]])
        seen = self.db.rows.get((beacon, self.filters["counter"]))
        return SimpleNamespace(data={"first_seen_at": seen} if seen else None)


class FreshnessTests(unittest.TestCase):
    def setUp(self):
        self.db = Sightings()
        self.t0 = datetime(2026, 10, 27, 9, 0, tzinfo=timezone.utc)

    def fresh(self, counter, seconds):
        return broadcast_beacon.fresh(self.db, "b1", counter, now=self.t0 + timedelta(seconds=seconds))

    def test_whole_class_shares_one_code_within_the_window(self):
        self.assertTrue(self.fresh(100, 0))
        self.assertTrue(self.fresh(100, 10))
        self.assertTrue(self.fresh(100, 30))
        self.assertFalse(self.fresh(100, 31))          # replayed after the window

    def test_far_behind_newest_is_refused(self):
        self.assertTrue(self.fresh(100, 0))
        self.assertTrue(self.fresh(94, 1))             # 6 steps behind: a slow phone
        self.assertFalse(self.fresh(93, 1))            # an old recording never seen before

    def test_boards_are_separate(self):
        self.assertTrue(self.fresh(100, 0))
        self.assertTrue(broadcast_beacon.fresh(self.db, "b2", 5, now=self.t0))


class Database:
    def table(self, name):
        query = Mock()
        for method in ["select", "eq", "maybe_single"]:
            getattr(query, method).return_value = query
        data = {
            "sessions": {"id": "s1", "course_id": "c1", "is_open": True, "beacon_id": "b1",
                         "beacons": {"ble_room_code": "TEST-101"}},
            "course_enrollments": {"id": "enrollment"},
            "beacons": {"ble_secret": KEY},
        }.get(name)
        query.execute.return_value = SimpleNamespace(data=data)
        return query


class RouteTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(route, "supabase_admin", Database())
        patcher.start()
        self.addCleanup(patcher.stop)
        self.is_fresh = True
        fresh = patch.object(broadcast_beacon, "fresh", side_effect=lambda *a, **k: self.is_fresh)
        fresh.start()
        self.addCleanup(fresh.stop)
        self.web = Flask(__name__)
        self.web.config.update(SECRET_KEY="test", RATELIMIT_ENABLED=False,
                               PROXIMITY_RECEIPT_SECRET="r" * 64,
                               CHECKIN_PROXIMITY_METHOD="broadcast")
        app.limiter.init_app(self.web)
        self.web.register_blueprint(route.api_checkin_bp)
        self.client = self.web.test_client()
        with self.client.session_transaction() as session:
            session.update(user_id="student", user_role="student", csrf_token="csrf")

    def prove(self, room="TEST-101", **body):
        return self.client.post("/api/checkin/proximity",
                                json={"session_id": "s1", "room_code": room, **body},
                                headers={"X-CSRF-Token": "csrf"})

    def test_board_code_gets_a_receipt_bound_to_broadcast(self):
        response = self.prove(broadcast_payload=FIRMWARE_VECTOR)
        self.assertEqual(response.status_code, 200, response.json)
        token = response.json["proximity_receipt"]
        self.assertIsNone(proximity_receipt.verify(token, "r" * 64, "student", "s1", "broadcast", "TEST-101"))
        self.assertEqual(proximity_receipt.verify(token, "r" * 64, "student", "s1", "ble", "TEST-101"),
                         "proximity_receipt_invalid")

    def test_room_code_alone_is_refused(self):
        response = self.prove()
        self.assertEqual(response.status_code, 400)
        self.assertNotIn("proximity_receipt", response.json)

    def test_forged_or_stale_code_is_refused(self):
        forged = FIRMWARE_VECTOR[:-2] + "00"
        self.assertEqual(self.prove(broadcast_payload=forged).status_code, 400)
        self.is_fresh = False
        response = self.prove(broadcast_payload=FIRMWARE_VECTOR)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json["error_code"], "room_code_invalid")

    def test_wrong_room_is_refused(self):
        response = self.prove(room="ROOM-202", broadcast_payload=FIRMWARE_VECTOR)
        self.assertEqual(response.status_code, 400)
        self.assertNotIn("proximity_receipt", response.json)


if __name__ == "__main__":
    unittest.main()
