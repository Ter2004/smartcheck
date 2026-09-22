"""Reset SmartCheck application data and Auth accounts in one DB transaction.

Default: print the plan without connecting. Run with --apply and set
SMARTCHECK_TEST_PASSWORD to the agreed test password. Existing classroom
hardware configuration is restored; biometric/attendance/history data is cleared.
Storage files are outside this database-only reset.
"""
import argparse
import os
from pathlib import Path
from uuid import uuid4

import psycopg2
from psycopg2 import sql
from psycopg2.extras import Json
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
ACCOUNTS = [(f"std{i}", "student") for i in range(1, 5)] + [
    (f"t{i}", "teacher") for i in range(1, 6)
] + [("admin", "admin")]
TABLES = (
    "attendance", "audit_logs", "consent_logs", "student_biometrics",
    "course_enrollments", "schedules", "sessions", "courses", "users",
    "flask_sessions", "beacons",
)


def reset(connection, password):
    if len(password) < 6:
        raise ValueError("Test password must contain at least 6 characters")
    with connection:
        with connection.cursor() as cursor:
            cursor.execute("SET LOCAL lock_timeout = '10s'")
            cursor.execute("SET LOCAL statement_timeout = '60s'")
            cursor.execute("SELECT n.nspname FROM pg_extension e JOIN pg_namespace n "
                           "ON n.oid=e.extnamespace WHERE e.extname='pgcrypto'")
            crypto = cursor.fetchone()
            if not crypto:
                raise RuntimeError("pgcrypto is required")
            # Keep hardware identifiers so the physical classroom still works.
            cursor.execute("CREATE TEMP TABLE saved_beacons ON COMMIT DROP AS "
                           "SELECT * FROM public.beacons")
            cursor.execute("SELECT id FROM saved_beacons WHERE is_active "
                           "ORDER BY room_name, id LIMIT 1")
            beacon = cursor.fetchone()
            if not beacon:
                raise RuntimeError("An active classroom beacon is required before reset")
            cursor.execute(sql.SQL("TRUNCATE {} RESTART IDENTITY").format(
                sql.SQL(', ').join(sql.Identifier('public', table) for table in TABLES)))
            # Auth child rows (identities, sessions, refresh tokens) cascade.
            cursor.execute("DELETE FROM auth.users")
            cursor.execute("DELETE FROM auth.audit_log_entries")
            cursor.execute("INSERT INTO public.beacons SELECT * FROM saved_beacons")
            ids = {}
            for alias, role in ACCOUNTS:
                uid = str(uuid4())
                ids[alias] = uid
                email = f"{alias}@smartcheck.local"
                cursor.execute(sql.SQL("""
                    INSERT INTO auth.users
                      (instance_id, id, aud, role, email, encrypted_password,
                       email_confirmed_at, confirmation_token, recovery_token,
                       email_change_token_new, email_change, raw_app_meta_data,
                       raw_user_meta_data, created_at, updated_at)
                    VALUES ('00000000-0000-0000-0000-000000000000', %s,
                            'authenticated', 'authenticated', %s,
                            {}.crypt(%s, {}.gen_salt('bf')), now(), '', '', '', '',
                            %s, %s, now(), now())
                    """).format(sql.Identifier(crypto[0]), sql.Identifier(crypto[0])),
                    (uid, email, password,
                     Json({'provider': 'email', 'providers': ['email']}),
                     Json({'full_name': alias})))
                cursor.execute("""
                    INSERT INTO auth.identities
                      (provider_id, user_id, identity_data, provider, created_at, updated_at)
                    VALUES (%s, %s, %s, 'email', now(), now())
                    """, (uid, uid, Json({'sub': uid, 'email': email,
                                         'email_verified': True, 'phone_verified': False})))
                cursor.execute("""
                    INSERT INTO public.users
                      (id, email, full_name, role, student_id, is_active, must_change_password)
                    VALUES (%s, %s, %s, %s, %s, true, false)
                    """, (uid, email, alias, role, alias if role == 'student' else None))
            for day in range(5):
                course_id = str(uuid4())
                cursor.execute("""
                    INSERT INTO public.courses
                      (id, code, name, teacher_id, semester, section, is_active)
                    VALUES (%s, %s, %s, %s, '1', '001', true)
                    """, (course_id, f'TEST{day + 1}', f'Test class {day + 1}', ids[f't{day + 1}']))
                cursor.execute("""
                    INSERT INTO public.schedules
                      (course_id, day_of_week, start_time, end_time, beacon_id)
                    VALUES (%s, %s, '10:30', '22:00', %s)
                    """, (course_id, day, beacon[0]))
                for student in range(1, 5):
                    cursor.execute("INSERT INTO public.course_enrollments (course_id, student_id) "
                                   "VALUES (%s, %s)", (course_id, ids[f'std{student}']))
            for table, expected in [('users', 10), ('courses', 5), ('schedules', 5),
                                    ('course_enrollments', 20), ('attendance', 0),
                                    ('student_biometrics', 0), ('flask_sessions', 0)]:
                cursor.execute(sql.SQL('SELECT count(*) FROM public.{}').format(sql.Identifier(table)))
                if cursor.fetchone()[0] != expected:
                    raise RuntimeError(f'Unexpected count in {table}; rolling back')
            cursor.execute('SELECT count(*) FROM auth.users')
            if cursor.fetchone()[0] != 10:
                raise RuntimeError('Unexpected Auth user count; rolling back')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='Apply destructive reset')
    args = parser.parse_args()
    print('10 accounts: std1-std4, t1-t5, admin; 5 courses; 20 enrollments.')
    print('Monday t1 through Friday t5, each 10:30-22:00 Asia/Bangkok.')
    print('Clear old application data and Auth accounts; restore classroom hardware configuration.')
    if not args.apply:
        print('Plan only: database was not accessed.')
        return
    load_dotenv(ROOT / '.env')
    password = os.environ.get('SMARTCHECK_TEST_PASSWORD', '')
    if len(password) < 6:
        raise SystemExit('Set SMARTCHECK_TEST_PASSWORD to the agreed password (at least 6 characters).')
    connection = psycopg2.connect(os.environ['DATABASE_URL'], connect_timeout=10)
    try:
        reset(connection, password)
    finally:
        connection.close()
    print('Reset committed and row counts verified.')


if __name__ == '__main__':
    main()
