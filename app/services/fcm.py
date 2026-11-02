"""Firebase Cloud Messaging (HTTP v1) sender for the Android app's notifications.

Configured by FCM_SERVICE_ACCOUNT_FILE: the service-account JSON downloaded from
Firebase (Project settings → Service accounts → Generate new private key). Keep
it out of git (instance/ is ignored). Without it, push is off and nothing is sent.

The server signs a short JWT with the service account's key, trades it for an
OAuth access token (cached ~1 hour), then posts one message per device token.
docs/push-notifications.md has the setup steps.
"""
import json
import logging
import threading
import time

import httpx
import jwt

_log = logging.getLogger("smartcheck.fcm")

TOKEN_URL = "https://oauth2.googleapis.com/token"
SCOPE = "https://www.googleapis.com/auth/firebase.messaging"
SEND_URL = "https://fcm.googleapis.com/v1/projects/{project}/messages:send"
TIMEOUT_S = 10

SENT, INVALID_TOKEN, FAILED = "sent", "invalid_token", "failed"


class Sender:
    def __init__(self, service_account, client=None):
        self._account = service_account
        self._client = client or httpx.Client(timeout=TIMEOUT_S)
        self._lock = threading.Lock()
        self._token, self._token_until = None, 0.0

    @property
    def project(self):
        return self._account["project_id"]

    def _access_token(self):
        with self._lock:
            if self._token and time.time() < self._token_until:
                return self._token
            now = int(time.time())
            assertion = jwt.encode({"iss": self._account["client_email"], "scope": SCOPE,
                                    "aud": TOKEN_URL, "iat": now, "exp": now + 3600},
                                   self._account["private_key"], algorithm="RS256")
            r = self._client.post(TOKEN_URL, data={
                "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer", "assertion": assertion})
            r.raise_for_status()
            body = r.json()
            self._token = body["access_token"]
            self._token_until = time.time() + int(body.get("expires_in", 3600)) - 300
            return self._token

    def send(self, device_token, message):
        """Send one message (FCM v1 'message' without 'token'); SENT, INVALID_TOKEN or FAILED."""
        try:
            r = self._client.post(SEND_URL.format(project=self.project),
                                  headers={"Authorization": "Bearer " + self._access_token()},
                                  json={"message": {**message, "token": device_token}})
        except Exception as e:
            _log.warning("[FCM] send failed: %s", type(e).__name__)
            return FAILED
        if r.status_code == 200:
            return SENT
        # 404 UNREGISTERED: app uninstalled or token rotated. 400 INVALID_ARGUMENT
        # names the token when it is malformed. Both mean: forget this token.
        status = ""
        try:
            status = r.json().get("error", {}).get("status", "")
        except ValueError:
            pass
        if r.status_code == 404 or status == "UNREGISTERED" or (
                r.status_code == 400 and status == "INVALID_ARGUMENT" and "token" in r.text.lower()):
            return INVALID_TOKEN
        _log.warning("[FCM] send refused: HTTP %s %s", r.status_code, status)
        return FAILED


_sender = None
_sender_lock = threading.Lock()


def sender(config):
    """The configured Sender, or None when FCM_SERVICE_ACCOUNT_FILE is unset or unreadable."""
    global _sender
    path = config.get("FCM_SERVICE_ACCOUNT_FILE") or ""
    if not path:
        return None
    with _sender_lock:
        if _sender is None:
            try:
                with open(path, encoding="utf-8") as f:
                    account = json.load(f)
                _sender = Sender(account)
            except (OSError, ValueError, KeyError) as e:
                _log.error("[FCM] cannot read FCM_SERVICE_ACCOUNT_FILE: %s", type(e).__name__)
                return None
        return _sender
