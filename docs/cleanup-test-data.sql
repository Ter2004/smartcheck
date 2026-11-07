-- SmartCheck: remove leftover test data. Run in the Supabase SQL Editor yourself
-- (Claude never deletes data). Read each block, run the SELECT first, then the DELETE.

-- 1) The test timetable slot "TEST1, Friday 00:00-05:00" (from the 5-6 Nov phone tests).
--    While it exists, the scheduler creates a TEST1 class at midnight every Friday,
--    which shows up on t1's home page. Checked 2026-11-08: id 048125b3-069b-41ec-a185-db88e611afc0.
select id, day_of_week, start_time, end_time
from schedules
where id = '048125b3-069b-41ec-a185-db88e611afc0';

delete from schedules
where id = '048125b3-069b-41ec-a185-db88e611afc0';

-- 2) KEEP for now: the TEST1 classes and check-ins of 5-6 Nov 2026.
--    They are the source of the numbers in the report and the poster
--    (server check-in 0.83-1.10 s, 9-10 s per student, the 6 Nov round).
--    Delete them only after the report has been submitted:
-- select id, title, start_time, opened_at, end_time from sessions
-- where course_id = '24b8f530-a651-4acd-a5b1-fe6e900ac4c7' order by start_time;
-- delete from attendance where session_id in (
--     select id from sessions where course_id = '24b8f530-a651-4acd-a5b1-fe6e900ac4c7'
--     and start_time >= '2026-11-05' and start_time < '2026-11-07');
-- delete from sessions where course_id = '24b8f530-a651-4acd-a5b1-fe6e900ac4c7'
--     and start_time >= '2026-11-05' and start_time < '2026-11-07';

-- 3) After the field test and the report: remove the field-test students' face data
--    from the admin page (ข้อมูลใบหน้า > ถอนความยินยอม / ลบข้อมูลใบหน้า), as promised
--    in the consent form, then their accounts.
