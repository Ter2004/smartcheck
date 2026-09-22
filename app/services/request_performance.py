"""Request timing through response headers, including session load/save.

Status-query metrics cover enrollment status reads only, not all database I/O.
Never log URLs, user IDs, images, embeddings, cookies, or request bodies.
"""
import logging
from contextvars import ContextVar
from time import perf_counter

from flask import request

_metrics = ContextVar("request_performance", default=None)
_log = logging.getLogger("smartcheck.performance")


def execute_status_query(query):
    started = perf_counter()
    try:
        return query.execute()
    finally:
        metrics = _metrics.get()
        if metrics is not None:
            metrics["status_queries"] += 1
            metrics["status_query_ms"] += (perf_counter() - started) * 1000


def init_request_performance(app):
    @app.before_request
    def identify_route():
        metrics = _metrics.get()
        if metrics is not None:
            metrics["endpoint"] = request.endpoint or "unmatched"

    original = app.wsgi_app

    def timed_app(environ, start_response):
        started = perf_counter()
        metrics = dict(endpoint="unmatched", status_queries=0, status_query_ms=0.0)
        token = _metrics.set(metrics)

        def timed_start_response(status, headers, exc_info=None):
            elapsed = (perf_counter() - started) * 1000
            _log.info(
                "endpoint=%s status=%s response_ms=%.2f status_queries=%d status_query_ms=%.2f",
                metrics["endpoint"], status.split()[0], elapsed,
                metrics["status_queries"], metrics["status_query_ms"],
            )
            return start_response(status, headers, exc_info)

        try:
            return original(environ, timed_start_response)
        finally:
            _metrics.reset(token)

    app.wsgi_app = timed_app
