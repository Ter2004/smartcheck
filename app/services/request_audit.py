"""Request-local correlation and allowlisted check-in diagnostics."""
import json
import logging
import time
import uuid
import re
from contextlib import contextmanager
from functools import wraps
from flask import g, has_request_context, request, current_app


def request_id():
    if not has_request_context():
        return "background"
    if not getattr(g, "audit_request_id", None):
        g.audit_request_id = uuid.uuid4().hex
    return g.audit_request_id


class RequestLogger(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        endpoint = request.endpoint if has_request_context() else "background"
        return f"request_id={request_id()} endpoint={endpoint} {msg}", kwargs


logger = RequestLogger(logging.getLogger("smartcheck.checkin"), {})


def event(step, result, **details):
    logger.info("step=%s result=%s details=%s", step, result,
                json.dumps(details, ensure_ascii=True, separators=(",", ":")))


def reject(reason, **details):
    if "received" in details:
        value = details["received"]
        # Show stale frontend action names/empty strings, never arbitrary payloads.
        details["received"] = (value if value is None or
            (isinstance(value, str) and re.fullmatch(r"[a-z_]{0,32}", value))
            else "[redacted]")
    g.checkin_rejection = reason
    event(reason, "reject", **details)


@contextmanager
def stage(name):
    started = time.perf_counter()
    event(name, "start")
    try:
        yield
    except Exception as error:
        event(name, "error", exception_type=type(error).__name__)
        raise
    else:
        event(name, "complete", elapsed_ms=round((time.perf_counter() - started) * 1000, 2))


def audited_checkin(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        event("request", "start")
        try:
            response = current_app.make_response(func(*args, **kwargs))
        except Exception as error:
            from werkzeug.exceptions import HTTPException
            reason = f"http_{error.code}" if isinstance(error, HTTPException) else "unhandled_exception"
            reject(reason, exception_type=type(error).__name__)
            if isinstance(error, HTTPException):
                # Preserve Flask's existing error handlers, including rate limits.
                response = current_app.make_response(current_app.handle_http_exception(error))
                response.headers["X-Request-ID"] = request_id()
                event("request", "complete", status=response.status_code)
                return response
            raise
        if response.status_code >= 300 and not getattr(g, "checkin_rejection", None):
            reasons = {302: "authentication_or_role_required", 403: "csrf_rejected", 429: "rate_limited"}
            reject(reasons.get(response.status_code, "http_rejected"), status=response.status_code)
        event("request", "complete", status=response.status_code)
        response.headers["X-Request-ID"] = request_id()
        return response
    return wrapped
