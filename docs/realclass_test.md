# Real-class test (20-30 people)

This test answers the committee's questions with measurements from a real class:
- what room and class size the system was tested in;
- how long enrollment and check-in take;
- how often a genuine student is refused on the first try;
- whether check-in beats calling names.

The report comes from `scripts/realclass_report.py`.

## Before the day

1. **Apply the migration.** Run `database/migrations/20261019_face_stack_v2.sql`
   and its verification query, and keep the output.
2. **Choose the face settings.** Pick `FACE_CLAHE`, `ARCFACE_MATCH_THRESHOLD`,
   `ARCFACE_DUP_THRESHOLD` and `ENROLL_SELF_SIM_MIN` from
   `scripts/compare_models.py` on your own photos. Set them, plus
   `FACE_ENGINE=arcface` and `CHECKIN_LIVENESS=passive`, then start the server
   and keep the startup log lines.
3. **Prepare the room.** Have a scheduled session for the room (the scheduler
   opens it automatically) and the ESP32 or TOTP room proximity working.
4. **Consent.**
   - Participants enroll through the app's PDPA screen (version 1.2).
   - Tell them they can withdraw from the dashboard.
   - Use pseudonyms (S01…) in every sheet; real names stay off the sheets.

## On the day

**Record:**
- the room name;
- the seats, the class size and how many people took part;
- the phone models (at least iPhone vs Android counts);
- the lighting (daylight or fluorescent).

**Enrollment.** Each student enrolls on their own phone. An observer writes:
- `start`: when the student taps "เริ่มและเปิดกล้อง";
- `end`: when the result appears;
- the result, as success 1 or 0 with the message shown.

Every retry is a new row with `attempt` + 1.

**Roll call.** The teacher calls every name while an observer runs a stopwatch,
from the first name to the last answer. Write down the total seconds.

**Check-in.**
- All students check in at the same time when the session opens. An observer
  records each student's `start`, when the check-in page is ready, and `end`,
  when the result appears.
- Also note the clock time when the first student starts and when the last one
  succeeds.

**Server log.** Keep the log of the check-in window, i.e. the `[FACE] engine=arcface best=…` lines.

## Sheet template (`attempts.csv`)

```
student,phase,attempt,start,end,success,reason
S01,enroll,1,09:02:10,09:02:31,1,
S02,enroll,1,09:02:12,09:02:40,0,หันหน้าไปทางซ้ายให้มากขึ้นตามคำสั่ง
S02,enroll,2,09:02:45,09:03:02,1,
S01,checkin,1,09:30:05,09:30:12,1,
```

## Report

```
python scripts/realclass_report.py --attempts attempts.csv --rollcall-seconds 145 \
    --room "LAB 2" --class-size 28 --log checkin_window.log --json realclass_report.json
```

**What it reports:**

| item | details |
|---|---|
| enrollment | time per attempt and to success (median, p90); first-attempt success; failure reasons |
| check-in | time per student (median, p90); first-attempt reject rate; class elapsed from first start to last end |
| check-in vs roll call | the ratio of class elapsed to roll-call time |
| genuine scores | distribution against the threshold, how many fell below it, and the 5 lowest margins |

**Read it honestly:**
- One class is one room, one lighting setup and one group of phones. Say so.
- A score margin close to 0 means that student is at risk of being refused.
  List those students next to the threshold you chose.
- Times are observer stopwatch times, which include network time and the
  student's own reaction.
