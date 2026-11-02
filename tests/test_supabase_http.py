"""Supabase REST transport: a dropped read is retried once; writes are not; the hook survives rebuilds."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from app.services import supabase_http


class RetryTests(unittest.TestCase):
    def send(self, method, outcomes):
        transport = supabase_http._RetryDroppedReads()
        request = httpx.Request(method, "https://example.supabase.co/rest/v1/x")
        with patch.object(httpx.HTTPTransport, "handle_request", side_effect=outcomes) as sent:
            try:
                return transport.handle_request(request), sent.call_count
            except httpx.RemoteProtocolError:
                return None, sent.call_count

    def test_a_dropped_read_is_sent_again(self):
        response, calls = self.send("GET", [httpx.RemoteProtocolError("Server disconnected"), httpx.Response(200)])
        self.assertEqual((response.status_code, calls), (200, 2))

    def test_a_read_dropped_twice_still_fails(self):
        response, calls = self.send("GET", [httpx.RemoteProtocolError("x"), httpx.RemoteProtocolError("x")])
        self.assertEqual((response, calls), (None, 2))

    def test_writes_are_never_sent_twice(self):
        for method in ("POST", "PATCH", "DELETE"):
            with self.subTest(method=method):
                response, calls = self.send(method, [httpx.RemoteProtocolError("x"), httpx.Response(201)])
                self.assertEqual((response, calls), (None, 1))


class HardenTests(unittest.TestCase):
    def test_every_rebuilt_postgrest_client_gets_the_transport(self):
        built = []

        def build(**kwargs):
            postgrest = SimpleNamespace(session=httpx.Client(base_url="https://example.supabase.co/rest/v1",
                                                             headers={"apikey": "k"}, timeout=7))
            built.append(postgrest)
            return postgrest

        client = SimpleNamespace(_init_postgrest_client=build, _postgrest="old")
        supabase_http.harden(client)
        self.assertIsNone(client._postgrest, "rebuilt on next use")
        for _ in range(2):   # supabase-py rebuilds it after auth events
            postgrest = client._init_postgrest_client(rest_url="u")
            transport = postgrest.session._transport
            self.assertIsInstance(transport, supabase_http._RetryDroppedReads)
            self.assertEqual(transport._pool._keepalive_expiry, supabase_http.KEEPALIVE_SECONDS)
            self.assertEqual(postgrest.session.headers["apikey"], "k")
            self.assertEqual(postgrest.session.timeout.read, 7)
        self.assertEqual(len(built), 2)


if __name__ == "__main__":
    unittest.main()
