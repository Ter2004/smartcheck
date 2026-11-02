"""Supabase REST connections: kept warm longer, and a read the server dropped is retried.

Measured 2026-11-02 (this PC to Supabase ap-south-1):
- A query on a warm connection took ~120 ms. After more than 5 s idle (httpx's default
  keepalive), the next one opened a new connection and took 190-450 ms.
- Supabase sometimes closes a pooled HTTP/2 connection under a request
  ("Server disconnected", httpx.RemoteProtocolError). On a page that turned into a 500.
"""
import httpx

KEEPALIVE_SECONDS = 60


class _RetryDroppedReads(httpx.HTTPTransport):
    """Send a GET/HEAD once more, on a new connection, if the server dropped the old one.

    Writes are never retried: whether the server applied them is unknown.
    """

    def handle_request(self, request):
        try:
            return super().handle_request(request)
        except httpx.RemoteProtocolError:
            if request.method not in ("GET", "HEAD"):
                raise
            return super().handle_request(request)


def _session_like(old):
    return httpx.Client(
        base_url=old.base_url, headers=old.headers, timeout=old.timeout, follow_redirects=True,
        transport=_RetryDroppedReads(http2=True, limits=httpx.Limits(
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
