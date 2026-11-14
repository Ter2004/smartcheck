"""Supabase REST connections: kept warm longer, and a read the server dropped is retried.

Measured 2026-11-02 (this PC to Supabase ap-south-1):
- A query on a warm connection took ~120 ms. After more than 5 s idle (httpx's default
  keepalive), the next one opened a new connection and took 190-450 ms.
- Supabase sometimes closes a pooled HTTP/2 connection under a request
  ("Server disconnected", httpx.RemoteProtocolError). On a page that turned into a 500.

Measured 2026-11-10 (1,500 small reads, six at a time, as the pages now read):
- HTTP/2, one connection shared by the reading threads: 38-57 requests dropped by the
  server ("Server disconnected", GOAWAY protocol error), and 5-10 of them failed a
  second time and reached the page. One at a time: none in 300.
- HTTP/1.1, one connection per concurrent read: none dropped in 1,500, and no slower.
So the reads use HTTP/1.1. The single retry stays for a connection the server closed
while it sat idle.
"""
import httpx

KEEPALIVE_SECONDS = 60


class _RetryDroppedReads(httpx.HTTPTransport):
    """Send a GET/HEAD once more, on a new connection, if the server dropped the old one.

    Writes are never retried: whether the server applied them is unknown.
    """

    def handle_request(self, request):
        from time import perf_counter
        from app.services.request_performance import record_db_call
        started = perf_counter()
        try:
            return super().handle_request(request)
        except httpx.RemoteProtocolError:
            if request.method not in ("GET", "HEAD"):
                raise
            return super().handle_request(request)
        finally:
            record_db_call((perf_counter() - started) * 1000)   # no-op unless timing is on


def _session_like(old):
    return httpx.Client(
        base_url=old.base_url, headers=old.headers, timeout=old.timeout, follow_redirects=True,
        transport=_RetryDroppedReads(http2=False, limits=httpx.Limits(
            max_connections=100, max_keepalive_connections=20, keepalive_expiry=KEEPALIVE_SECONDS)))


def harden(client):
    """Give a supabase-py client's REST (PostgREST) calls the transport above.

    supabase-py rebuilds its PostgREST client after auth events, so the hook sits on
    the builder and every rebuilt client gets the same transport.
    """
    build = client._init_postgrest_client

    def init(*args, **kwargs):
        postgrest = build(*args, **kwargs)
        old = postgrest.session
        postgrest.session = _session_like(old)
        old.close()
        return postgrest

    client._init_postgrest_client = init
    client._postgrest = None
    return client
