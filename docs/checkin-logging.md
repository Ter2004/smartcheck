# Check-in diagnostics

Restart the server to load the logging changes. From the project directory:

```powershell
python -m unittest discover -s tests -v
python run.py 2>&1 | Tee-Object -FilePath checkin-demo.log
```

For an existing Docker deployment, rebuild/restart that deployment and read its
container logs instead; running a second local server does not update it.

Reproduce the failure and copy `X-Request-ID` from the `/api/checkin` response in
browser Network tools. Filter the log with:

```powershell
Select-String -Path checkin-demo.log -SimpleMatch 'request_id=PASTE_ID_HERE'
```

All 34 explicit route rejection branches emit a stable `step=<reason>
result=reject details={...}` event. The outer wrapper also records authentication,
role, CSRF and HTTP rejections. Logs identify the endpoint and a server-generated,
request-local ID; shared face-service logs, including COMBINED_SPOOF, use that same
ID in the same request. Different enrollment/passive requests receive different
IDs. Incoming client headers cannot choose the ID.

An empty action appears as `step=liveness_action_invalid result=reject
details={"received":""}`. Short lowercase action names are retained; other values
are redacted to avoid logging arbitrary payloads. Exceptions are logged by type,
not their message or request body.

TOTP logs `step=totp result=pass`, `wrong_or_stale`, or `verifier_error`.
The submitted six digits carry no timestamp: a wrong code and an expired code
cannot reliably be distinguished. We deliberately do not search old counters
and claim a coincidental six-digit match proves expiry. Acceptance remains ±1
window, and no raw codes or secrets are logged.

`device_lookup`, `biometrics_lookup`, `extraction`, and `matching` each emit
`start`, followed by `complete` with elapsed milliseconds or `error` with an
exception type. Session/enrollment/attendance lookups and anti-spoof also have
stage markers. A start without complete/error indicates an operation still
pending, an interrupted worker, or missing logs—not a validation rejection.

Validation: 33 tests pass, including shared spoof correlation, distinct request
IDs, empty-action logging, TOTP outcomes, extraction errors, CSRF rejection,
and checks that secrets/codes/exception payloads are absent from captured logs.
