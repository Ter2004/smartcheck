# P-4 — Enrollment status performance rollout

This change shares one status read per Flask request. It does not cache across
requests, change AI checks, or change session expiration. Existing deployments
work without a database migration; the compact RPC is separately enabled.

## Baseline before enabling optimizations

Deploy this code with:

```text
PERFORMANCE_LOG_ENABLED=true
ENROLLMENT_STATUS_CACHE=false
ENROLLMENT_STATUS_RPC=false
```

Collect `smartcheck.performance` logs on the actual application server. Exercise
dashboard, enrollment, checkin, and spoof_check with the same test account and
real capture flow. Separate first/cold requests from warm requests. Compare
median and p95 for at least 30 warm samples per route and record sample counts,
deployment region, concurrency, status codes, and whether enrollment is pending
or completed. Completed enrollment redirects early and is not comparable to a
rendered enrollment page. Do not substitute synthetic-image inference timings.

`response_ms` measures WSGI entry through response headers, including Flask
session load/save. It excludes body streaming, browser work, and network delivery.
`status_queries` and `status_query_ms` cover **only enrollment status reads**,
including failed executions; they are not total database query metrics. Route
times include other database calls and model work but do not separate those.
No user IDs, request bodies, images, or embeddings are logged.

## Enable and compare

1. Set `ENROLLMENT_STATUS_CACHE=true` and repeat the same measurements. Expected
   status-read counts: rendered enrollment 3 to 1; dashboard/checkin 2 to 1.
   Checkin can redirect early, in which case its count was already 1.
2. Apply `database/migrations/20260919_enrollment_status.sql` in Supabase SQL
   Editor. The function uses invoker permissions and only service_role may call
   it. Inspect existing data first:

   ```sql
   SELECT jsonb_typeof(face_embeddings) AS embedding_type, count(*)
   FROM public.student_biometrics GROUP BY 1;
   ```

   SQL NULL, JSON null, and empty arrays mean no embeddings. Other non-array JSON
   is rejected by the RPC, rather than silently bypassing enrollment protection.
3. Set `ENROLLMENT_STATUS_RPC=true`, restart, and repeat. The RPC returns only
   `is_enrolled` and `baseline_ear`; it does not transmit embeddings. Check pending,
   completed, missing-row, and withdrawn users, including API enrollment guards.
   Database failures keep the guard closed (503); there is no silent fallback.
   Dashboard, enrollment, and checkin pages return a standalone Thai retry page
   with HTTP 503 and no-store, without redirecting or querying status again.
   Guarded APIs retain their JSON 503 response.
4. Disable performance logging after comparison if no longer needed.

Rollback: set `ENROLLMENT_STATUS_RPC=false`; set `ENROLLMENT_STATUS_CACHE=false`
to restore repeated reads as well. The read-only RPC may remain installed.

Local mocked regression tests verify query counts, fresh state across requests,
failure handling, and timing isolation. They do **not** establish production
latency gains or validate the migration against a live database. No live baseline
has been collected as part of this code change.

## Validation scope

Original change: 14 passing tests, using the project venv and unittest, not pytest:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -p 'test_*enrollment*.py'
```

That selection consisted of 9 existing tests in
`test_automatic_sessions_and_enrollment.py` and 5 new tests in
`test_enrollment_performance.py`. The review follow-up adds 2 tests for page
failures (16 total). These counts do not include `test_scheduler_startup.py`.
They do not claim that the full suite passes, or that venv package versions
match production pins. The user separately reports a system-Python full-suite
baseline of 54 passing / 8 failing tests, with the same 8 checkin TOTP failures
reproduced at HEAD. That baseline is distinct from this targeted test run.

Pre-existing working-tree changes to `app/scheduler.py`,
`app/static/js/checkin_flow.js`, `app/static/js/enrollment_flow.js`, and
`tests/test_scheduler_startup.py` are outside P-4. No multi-worker scheduler
safety or static asset optimization is claimed by this change.
