import copy
import unittest
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch
from flask import Flask
import app
from app.services.session_policy import bounds, state, occurrence
from app.services.enrollment_policy import authorization
from app.routes import teacher, student, admin
from app import scheduler


class DB:
    def __init__(self, rows):
        self.rows = copy.deepcopy(rows)
        self.writes = []

    def table(self, name):
        return Query(self, name)


class Query:
    def __init__(self, db, name):
        self.db, self.name, self.filters = db, name, []
        self.single, self.change = False, None

    def select(self, *a, **k): return self
    def limit(self, *a): return self
    def order(self, *a, **k): return self
    def eq(self, key, value): self.filters.append(lambda row: row.get(key) == value); return self
    def gte(self, key, value):
        self.filters.append(lambda row: datetime.fromisoformat(row[key]) >= datetime.fromisoformat(value)); return self
    def maybe_single(self): self.single = True; return self
    def update(self, row): self.change = ('update', row); return self
    def insert(self, row): self.change = ('insert', row); return self
    def upsert(self, row, **kwargs): self.change = ('upsert', row); return self

    def execute(self):
        rows = self.db.rows.setdefault(self.name, [])
        selected = [row for row in rows if all(test(row) for test in self.filters)]
        if self.change:
            action, value = self.change
            self.db.writes.append((self.name, action, value))
            if action == 'update':
                for row in selected: row.update(value)
            elif action == 'insert' or not any(row['course_id'] == value['course_id'] and row['start_time'] == value['start_time'] for row in rows):
                row = dict(value)
                row.setdefault('id', str(len(rows)))
                rows.append(row)
                selected = [row]
        return SimpleNamespace(data=(selected[0] if selected else None) if self.single else selected)


class WindowTests(unittest.TestCase):
    def row(self):
        return occurrence({'id':'sch','course_id':'c','start_time':'09:00','end_time':'12:00'}, date(2026,9,14))

    def test_exact_open_late_close_and_stale_cache(self):
        row = self.row()
        opens, late, closes = bounds(row)
        self.assertEqual(state(row, opens-timedelta(microseconds=1)), 'pending')
        row['is_open'] = False
        self.assertEqual(state(row, opens), 'open')
        self.assertEqual(late, opens+timedelta(minutes=15))
        self.assertEqual(state(row, closes), 'closed')
        row['is_open'] = True
        self.assertEqual(state(row, closes+timedelta(seconds=1)), 'closed')

    def test_invalid_cancelled_and_legacy_fail_closed(self):
        row = self.row(); now = bounds(row)[0]
        for change in [{'end_time':None}, {'late_at':'invalid'}, {'checkin_duration':0},
                       {'checkin_duration':-1}, {'session_kind':'legacy'}]:
            self.assertEqual(state(dict(row, **change), now), 'unconfigured')
        self.assertEqual(state(dict(row, cancelled_at=now.isoformat()), now), 'cancelled')

    def test_overnight_and_early_open_cross_midnight(self):
        row = occurrence({'id':'s','course_id':'c','start_time':'23:00','end_time':'01:00'}, date(2026,9,14))
        self.assertEqual(bounds(row)[2]-bounds(row)[0], timedelta(hours=2))
        row = occurrence({'id':'s','course_id':'c','start_time':'00:05','end_time':'02:00','open_before_minutes':15}, date(2026,9,14))
        self.assertEqual(bounds(row)[0].astimezone(scheduler.TZ_THAI).date(), date(2026,9,13))

    def test_scheduler_preserves_occurrences_and_cancellations(self):
        fixed = datetime(2026,9,14,3,tzinfo=timezone.utc)  # Monday 10:00 Bangkok
        schedules = [dict(id='s1',course_id='c',day_of_week=0,start_time='09:00',end_time='10:00',beacon_id='b',is_active=True,courses={'id':'c','code':'C','is_active':True}),
                     dict(id='s2',course_id='c',day_of_week=0,start_time='11:00',end_time='12:00',beacon_id='b',is_active=True,courses={'id':'c','code':'C','is_active':True})]
        db = DB({'schedules':schedules})
        with patch.object(scheduler, '_get_supabase', return_value=db), patch.object(scheduler, 'datetime') as clock:
            clock.now.return_value = fixed
            scheduler.auto_manage_sessions()
            count = len(db.rows['sessions'])
            second = next(row for row in db.rows['sessions'] if row['schedule_id']=='s2')
            second['cancelled_at'] = fixed.isoformat()
            original = copy.deepcopy(second)
            scheduler.auto_manage_sessions()
        self.assertEqual(len(db.rows['sessions']), count)
        self.assertEqual(second, original)
        today = [row for row in db.rows['sessions'] if row['start_time'].startswith('2026-09-14')]
        self.assertEqual(len(today), 2)
        self.assertNotEqual(today[0]['start_time'], today[1]['start_time'])


class AuthorizationTests(unittest.TestCase):
    def setUp(self):
        self.web = Flask(__name__)
        self.web.config.update(SECRET_KEY='test', RATELIMIT_ENABLED=False)
        app.limiter.init_app(self.web)
        self.web.add_url_rule('/login', 'auth.login', lambda: 'login')
        self.web.register_blueprint(teacher.teacher_bp, url_prefix='/teacher')
        self.web.register_blueprint(student.student_bp, url_prefix='/student')
        self.web.register_blueprint(admin.admin_bp, url_prefix='/admin')
        self.client = self.web.test_client()

    def login(self, role):
        with self.client.session_transaction() as sess:
            sess.update(user_id=role, user_role=role, policy_version=2, csrf_token='token')

    def test_enrollment_authorization_is_one_use_and_survives_withdrawal(self):
        db = DB({'users':[{'id':'student','face_enrolled_once':False}]})
        self.assertTrue(authorization(db, 'student'))
        db.rows['users'][0]['face_enrolled_once'] = True
        self.assertFalse(authorization(db, 'student'))
        db.rows['face_change_requests'] = [{'user_id':'student','status':'approved'}]
        self.assertTrue(authorization(db, 'student'))
        db.rows['face_change_requests'][0]['status'] = 'consumed'
        self.assertFalse(authorization(db, 'student'))

    def test_retired_teacher_and_student_endpoints(self):
        self.login('teacher')
        for url in ['/teacher/session/create','/teacher/session/x/toggle']:
            self.assertEqual(self.client.post(url, data={'csrf_token':'token'}).status_code,410)
        self.login('student')
        self.assertEqual(self.client.post('/student/api/self_verify', json={}, headers={'X-CSRF-Token':'token'}).status_code,410)

    def test_direct_reenrollment_is_denied_before_models(self):
        self.login('student')
        db = DB({'users':[{'id':'student','face_enrolled_once':True}]})
        with patch.object(student, 'supabase_admin', db):
            response = self.client.post('/student/api/enroll',json={'face_images':[]},headers={'X-CSRF-Token':'token'})
        self.assertEqual(response.status_code,403)
        self.assertFalse(db.writes)

    def test_override_requires_reason_and_roster(self):
        self.login('teacher')
        db = DB({'sessions':[{'id':'s','course_id':'c','courses':{'teacher_id':'teacher'}}]})
        with patch.object(teacher, 'supabase_admin', db):
            response = self.client.post('/teacher/session/s/override',data={'csrf_token':'token','student_id':'outsider','status':'present','reason':''})
            self.assertEqual(response.status_code,400)
            response = self.client.post('/teacher/session/s/override',data={'csrf_token':'token','student_id':'outsider','status':'present','reason':'Verified with instructor'})
            self.assertEqual(response.status_code,403)
        self.assertFalse(db.writes)

    def test_old_sessions_and_test_accounts_cannot_bypass_login_policy(self):
        self.login('student')
        with self.client.session_transaction() as sess: sess.pop('policy_version')
        self.assertEqual(self.client.get('/student/dashboard').status_code,302)
        self.login('student')
        with self.client.session_transaction() as sess: sess['is_test_account'] = True
        self.assertEqual(self.client.get('/student/dashboard').status_code,403)

    def test_student_cannot_review_own_request(self):
        self.login('student')
        response = self.client.post('/admin/face-change-requests/x/review',data={'csrf_token':'token','decision':'approved','reason':'Approve myself'})
        self.assertEqual(response.status_code,302)


if __name__ == '__main__': unittest.main()
