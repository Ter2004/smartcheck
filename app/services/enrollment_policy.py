"""Enrollment authorization is rechecked by a database trigger at commit."""
def authorization(db, user_id):
    user = db.table('users').select('face_enrolled_once').eq('id', user_id).maybe_single().execute().data
    if user is None:
        return False
    if not user.get('face_enrolled_once'):
        return True
    rows = db.table('face_change_requests').select('id').eq('user_id', user_id).eq('status', 'approved').limit(1).execute().data
    return bool(rows)
