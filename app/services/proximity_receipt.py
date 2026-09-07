"""Short-lived server-issued proximity receipts; never log their contents."""
import hashlib
import hmac
import os
from itsdangerous import URLSafeTimedSerializer, BadData, SignatureExpired

TTL_SECONDS = 90


def load_secret():
    value = os.getenv('PROXIMITY_RECEIPT_SECRET')
    if not value or len(value) < 32:
        raise RuntimeError('PROXIMITY_RECEIPT_SECRET must be set (at least 32 characters); startup aborted')
    return value


def serializer(secret):
    if not isinstance(secret, str) or len(secret) < 32:
        raise RuntimeError('Invalid PROXIMITY_RECEIPT_SECRET')
    return URLSafeTimedSerializer(secret, salt='smartcheck-proximity-v1',
                                  signer_kwargs={'digest_method': hashlib.sha256})


def binding(secret, student, session_id, method, room):
    # The receipt contains a keyed commitment, not the six-digit code itself.
    digest = hmac.new(secret.encode(), room.encode(), hashlib.sha256).hexdigest()
    return {'student': student, 'session': session_id, 'method': method, 'room': digest}


def issue(secret, student, session_id, method, room):
    return serializer(secret).dumps(binding(secret, student, session_id, method, room))


def verify(token, secret, student, session_id, method, room):
    signer = serializer(secret)
    if not token:
        return 'proximity_receipt_missing'
    if not isinstance(token, str) or not isinstance(room, str):
        return 'proximity_receipt_invalid'
    try:
        value = signer.loads(token, max_age=TTL_SECONDS)
    except SignatureExpired:
        return 'proximity_receipt_expired'
    except BadData:
        return 'proximity_receipt_invalid'
    if value != binding(secret, student, session_id, method, room):
        return 'proximity_receipt_invalid'
    return None
