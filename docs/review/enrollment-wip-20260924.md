# Enrollment WIP review — 2026-09-24

Status: reviewed locally; enrollment commit awaits owner approval as requested
on the SmartCheck Board. The changes are already present in the working tree.

## Changes reviewed

- `app/routes/student.py`: only the consent and withdrawal audit version changes
  from `1.0` to `1.1`. The pending diff does not modify `/api/enroll`.
- `app/templates/student/enroll_face.html`: describes stored face photographs,
  512-dimensional embeddings and measured EAR. This matches the existing upload
  of an enrollment image to the `face-images` bucket; previously the text claimed
  only numerical data was stored. Existing consent-log rows are not migrated.
- `app/services/face_service.py`: serializes DeepFace calls within a process using
  an RLock and guards the shared OpenCV cascade in both crop and temporal paths.
  Native OpenCV error diagnostics include code/function, not image contents.
- `tests/test_face_concurrency.py`: checks simultaneous extraction/inference,
  lock release after exceptions, and concurrent uses of the shared cascade.

No blocking defect was found in this diff. Locking serializes native model calls,
so requests may queue during simultaneous enrollment/check-in. The tests verify
mutual exclusion, not a production throughput or latency target. Each process has
its own model instances and locks; this does not coordinate multiple processes.

## Existing issue outside this diff

Withdrawal clears `face_image_url` in the biometrics row but does not remove the
uploaded object from the storage bucket. The pending text directs deletion
requests to administrators; this review does not establish legal compliance or
claim that automatic withdrawal removes the stored photograph. Storage deletion
needs a separate change if full automatic photo erasure is required.

## Validation

The concurrency tests and related face/enrollment tests are run locally with
mocked database operations. The calibration smoke run also exercises real
FaceNet512 extraction using existing local test images, without a database write.
Live enrollment on phones and production load remain separate board tasks.

The full Python suite also exposes 19 existing error outcomes in
`test_checkin_totp.py`: its database mock does not support the current attendance
lookup chain. Running that file against an isolated export of HEAD reproduced
the same 19 errors before these changes. They must not be reported as passing.

Only these four enrollment files should be included in its eventual commit;
the capture tests, documentation and calibration script are separate changes.
