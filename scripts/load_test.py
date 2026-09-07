"""
scripts/load_test.py
=====================
Load test เต็มรูปแบบสำหรับ /api/checkin — 50 นักศึกษาที่ seed ไว้ด้วย
scripts/seed_load_test.py

**สำคัญที่สุด — อ่านก่อนถ่ายภาพเทส:**
ไล่ pipeline ของ /api/checkin (app/routes/api_checkin.py) จริงแล้วยืนยันว่า
Moiré FFT / Screen-Texture FFT / Temporal-Variance / MiniFASNet (4 ชั้น
anti-spoof) ทำงาน "ก่อน" extract_embedding() (DeepFace.represent, ~350ms)
เสมอ — ถ้าภาพถูกตัดสินว่าเป็น spoof จะ return 400 "spoof": true ทันที
โดยไม่เรียก DeepFace เลย ดังนั้น:

  1. ต้องเป็นภาพถ่ายจริงจากกล้อง ไม่ใช่ภาพถ่ายจากหน้าจอ/สิ่งพิมพ์
     (จะโดน Moiré/Texture reject ก่อนถึง DeepFace)
  2. ต้องส่ง >= 2 เฟรมที่ "แตกต่างกันจริง" ใน face_images — ถ้าใช้ไฟล์
     เดียวกันซ้ำ 2-3 ครั้ง Temporal-Variance จะเห็น std=0 แล้วตัดสินว่าเป็น
     "ภาพนิ่ง" (spoof) ทันที ทั้งที่เป็นภาพจริง — ต้องเป็น burst shot
     คนละเฟรมจริง ๆ (ขยับหน้าเล็กน้อยระหว่างถ่ายก็พอ)
  3. ข้อกำหนดของ server_validate_frame (face_service.py:744-823):
     JPEG เท่านั้น, ขนาดไฟล์ 3KB-500KB, ความละเอียด 160x120 ถึง 1920x1080,
     Laplacian blur variance >= 8, สี channel std-dev >= 2.0 (ไม่ใช่ภาพสีเดียว)

ดูรายละเอียดเต็มใน docs/review/09-load-test-prep.md

Usage:
    # เฟส A — login ทั้ง 50 คน (ช้า ๆ, <=8/min), เก็บ session cookies + csrf token
    python scripts/load_test.py login

    # เฟส B — ยิง /api/checkin พร้อมกันทั้ง 50 request, บันทึกผล
    python scripts/load_test.py checkin --images photo1.jpg photo2.jpg photo3.jpg --label 1worker

    # ทำทั้งสองเฟสรวด (login แล้วต่อด้วย checkin ทันที)
    python scripts/load_test.py all --images photo1.jpg photo2.jpg --label 4workers

    # เทียบผล 2 รอบ (เช่น 1 worker vs 4 workers)
    python scripts/load_test.py compare results/1worker_*.json results/4workers_*.json

    # smoke test — login แค่ loadtest001 คนเดียว ยิง /api/checkin 1 ครั้ง
    # พิมพ์ status/body/elapsed เต็ม ๆ แล้ววินิจฉัยว่าตกที่ขั้นไหนของ pipeline
    # ใช้เช็คก่อนรัน load test 50 คนจริง ว่า deploy/salt/seed พร้อมหรือยัง
    python scripts/load_test.py smoke --images photo1.jpg photo2.jpg

    # warmup — ยิง /api/checkin ทีละ request (sequential) ให้ DeepFace/Fasnet
    # โหลดโมเดลจนครบก่อนวัด burst จริง — default count = 3 x --workers
    # (ดู docs/review/09-load-test-prep.md §3 สำหรับ runbook เต็ม รวมวิธีเช็ค
    # RAM plateau บน Railway metrics tab — elapsed variance อย่างเดียวไม่พอ)
    python scripts/load_test.py warmup --workers 4 --images photo1.jpg photo2.jpg

ต้องระบุ URL ของ server ที่จะยิงใส่ (Railway deployment เท่านั้น — gunicorn
รันบน Windows ไม่ได้ ห้ามยิงใส่ localhost บนเครื่องนี้):
    python scripts/load_test.py all --url https://xxxxx.up.railway.app --images ...

session_id ของ session ที่เปิดอยู่ (จาก seed_load_test.py) หาอัตโนมัติผ่าน
Supabase (อ่าน .env เดียวกับ seed script) — หรือระบุเองด้วย --session-id
"""
import argparse
import glob as globmod
import json
import os
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ============================================================
# ค่าคงที่ — ต้องตรงกับ scripts/seed_load_test.py
# ============================================================
LOADTEST_PASSWORD  = "LoadTest2026!"
STUDENT_COUNT      = 50
STUDENT_EMAIL_FMT  = "loadtest{:03d}@smartcheck.local"
COURSE_CODE        = "LOADTEST101"

LOGIN_MAX_PER_MIN  = 8          # ต่ำกว่า limit จริง 10/min (auth.py:54) ไว้เผื่อ
LOGIN_INTERVAL_SEC = 60.0 / LOGIN_MAX_PER_MIN
LOGIN_MAX_RETRIES  = 8          # ต่อ 429 หนึ่งคน — backoff แล้ว retry ไม่ fail ทิ้ง

CHECKIN_TIMEOUT_SEC = 30.0      # timeout ต่อ request /api/checkin (DeepFace เดี่ยว ๆ ~1-2s ปกติ)
WARMUP_TIMEOUT_SEC  = 90.0      # request แรก ๆ ของ worker ที่ยังไม่ warm ต้องโหลด TF/DeepFace/Fasnet เข้า
                                 # memory ครั้งแรก อาจช้ากว่า CHECKIN_TIMEOUT_SEC ปกติมาก — กันไว้ก่อน

STATE_FILE   = os.path.join(os.path.dirname(__file__), "load_test_state.json")
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "load_test_results")


# ============================================================
# Helpers — session state (cookies + csrf token ต่อคน)
# ============================================================

def _save_state(sessions: dict):
    """sessions: {email: {"cookies": {...}, "csrf_token": "..."}}"""
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(sessions, f, ensure_ascii=False, indent=2)


def _load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {}
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _session_from_state(email: str, state: dict) -> requests.Session:
    entry = state[email]
    s = requests.Session()
    s.cookies.update(entry["cookies"])
    return s


# ============================================================
# เฟส A — Login ทีละคน, pace <= 8/min, backoff+retry บน 429
# ============================================================

def _login_one(base_url: str, email: str) -> dict | None:
    """คืน {"cookies": dict, "csrf_token": str} หรือ None ถ้า login ไม่สำเร็จเลยหลัง retry ครบ"""
    s = requests.Session()

    attempt = 0
    backoff = 5.0
    while attempt <= LOGIN_MAX_RETRIES:
        attempt += 1
        try:
            resp = s.post(
                f"{base_url}/login",
                data={"email": email, "password": LOADTEST_PASSWORD},
                timeout=15,
                allow_redirects=True,
            )
        except requests.RequestException as e:
            print(f"  [{email}] network error: {e} — retry in {backoff:.0f}s")
            time.sleep(backoff)
            backoff = min(backoff * 1.5, 60)
            continue

        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else backoff
            print(f"  [{email}] 429 rate-limited — backoff {wait:.0f}s (attempt {attempt}/{LOGIN_MAX_RETRIES})")
            time.sleep(wait)
            backoff = min(backoff * 1.5, 60)
            continue

        # login.html แสดง flash "อีเมลหรือรหัสผ่านไม่ถูกต้อง" ถ้า fail แต่ status ยังเป็น 200
        # เช็คจาก final URL แทน — ถ้ายังอยู่ที่ /login แปลว่า login ไม่ผ่าน
        if resp.url.rstrip("/").endswith("/login"):
            print(f"  [{email}] login failed — ยังอยู่หน้า /login (เช็ค password/seed data)")
            return None

        m = re.search(r'<meta name="csrf-token" content="([^"]*)"', resp.text)
        csrf_token = m.group(1) if m else ""
        if not csrf_token:
            print(f"  [{email}] login สำเร็จแต่หา csrf-token ไม่เจอในหน้าที่ได้กลับมา (url={resp.url})")
            return None

        return {
            "cookies": requests.utils.dict_from_cookiejar(s.cookies),
            "csrf_token": csrf_token,
        }

    print(f"  [{email}] ให้ retry {LOGIN_MAX_RETRIES} ครั้งแล้วยังไม่สำเร็จ — ข้าม")
    return None


def phase_login(base_url: str):
    print(f"=== เฟส A: Login {STUDENT_COUNT} คน (<= {LOGIN_MAX_PER_MIN}/min) ===\n")
    state = {}
    failed = []

    for i in range(1, STUDENT_COUNT + 1):
        email = STUDENT_EMAIL_FMT.format(i)
        t0 = time.perf_counter()
        result = _login_one(base_url, email)
        elapsed = time.perf_counter() - t0

        if result:
            state[email] = result
            print(f"[{i:2d}/{STUDENT_COUNT}] {email} OK ({elapsed:.2f}s)")
        else:
            failed.append(email)
            print(f"[{i:2d}/{STUDENT_COUNT}] {email} FAILED")

        # pace ให้ไม่เกิน LOGIN_MAX_PER_MIN — เว้นตัวสุดท้ายไม่ต้องรอ
        if i < STUDENT_COUNT:
            remaining_sleep = LOGIN_INTERVAL_SEC - elapsed
            if remaining_sleep > 0:
                time.sleep(remaining_sleep)

    _save_state(state)
    print(f"\nDone — login สำเร็จ {len(state)}/{STUDENT_COUNT}, ล้มเหลว {len(failed)}")
    if failed:
        print(f"  ล้มเหลว: {', '.join(failed)}")
    print(f"บันทึก session state ไว้ที่ {STATE_FILE} (เฟส checkin จะอ่านไฟล์นี้)")
    return state


# ============================================================
# เตรียมภาพทดสอบ
# ============================================================

def _load_images(paths: list[str]) -> list[str]:
    """อ่านไฟล์ภาพ, เช็คเงื่อนไขคร่าว ๆ ตาม server_validate_frame, คืน list ของ data-URL base64"""
    if len(paths) < 2:
        print("ERROR: ต้องมีภาพอย่างน้อย 2 ไฟล์ (Temporal-Variance check ต้องการ >=2 เฟรม)", file=sys.stderr)
        sys.exit(1)

    raw_list = []
    for p in paths:
        if not os.path.exists(p):
            print(f"ERROR: ไม่พบไฟล์ {p}", file=sys.stderr)
            sys.exit(1)
        with open(p, "rb") as f:
            raw = f.read()

        size_kb = len(raw) / 1024
        if raw[:3] != b"\xff\xd8\xff":
            print(f"ERROR: {p} ไม่ใช่ JPEG (magic bytes ไม่ตรง) — server_validate_frame จะ reject", file=sys.stderr)
            sys.exit(1)
        if not (3 <= size_kb <= 500):
            print(f"WARNING: {p} ขนาด {size_kb:.1f}KB อยู่นอกช่วง 3-500KB ที่ server ยอมรับ (server_validate_frame)")
        raw_list.append(raw)

    # เช็คว่าไฟล์ไม่ซ้ำกันเป๊ะ (กับดัก Temporal-Variance: เฟรมเหมือนกัน = "ภาพนิ่ง")
    for i in range(len(raw_list)):
        for j in range(i + 1, len(raw_list)):
            if raw_list[i] == raw_list[j]:
                print(
                    f"WARNING: {paths[i]} และ {paths[j]} เป็นไฟล์เดียวกันไบต์ต่อไบต์ — "
                    "Temporal-Variance check (api_checkin.py:213-253) จะเห็น std=0 แล้วตัดสินว่า "
                    "'ตรวจพบภาพนิ่ง' (spoof=True) ทันที ทั้งที่เป็นภาพจริง ต้องใช้ burst shot คนละเฟรม"
                )

    import base64
    return ["data:image/jpeg;base64," + base64.b64encode(raw).decode() for raw in raw_list]


# ============================================================
# หา session_id อัตโนมัติจาก Supabase (course LOADTEST101, is_open=True)
# ============================================================

def _discover_session_id() -> str:
    from supabase import create_client
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key:
        print("ERROR: หา --session-id อัตโนมัติไม่ได้ (SUPABASE_URL/SUPABASE_SERVICE_KEY ไม่มีใน .env) "
              "ระบุ --session-id เองแทน", file=sys.stderr)
        sys.exit(1)
    sb = create_client(url, key)
    course = sb.table("courses").select("id").eq("code", COURSE_CODE).execute()
    if not course.data:
        print(f"ERROR: ไม่พบ course {COURSE_CODE} — รัน scripts/seed_load_test.py ก่อน", file=sys.stderr)
        sys.exit(1)
    course_id = course.data[0]["id"]
    sess = (
        sb.table("sessions").select("id, title")
        .eq("course_id", course_id).eq("is_open", True)
        .order("start_time", desc=True).limit(1)
        .execute()
    )
    if not sess.data:
        print(f"ERROR: ไม่พบ open session ของ {COURSE_CODE} — รัน scripts/seed_load_test.py ก่อน", file=sys.stderr)
        sys.exit(1)
    print(f"พบ session อัตโนมัติ: {sess.data[0]['title']} ({sess.data[0]['id']})")
    return sess.data[0]["id"]


# ============================================================
# เฟส B — ยิง /api/checkin พร้อมกันทั้งหมด
# ============================================================

def _checkin_one(base_url: str, email: str, entry: dict, session_id: str, images: list[str],
                  timeout: float = CHECKIN_TIMEOUT_SEC) -> dict:
    s = requests.Session()
    s.cookies.update(entry["cookies"])

    payload = {
        "session_id":      session_id,
        "ble_rssi":        -60,
        "ble_skip":        False,
        "liveness_action": "passive",     # ข้าม strict EAR blink check (api_checkin.py:153) —
                                           # ตั้งใจ: load test วัด compute cost ของ spoof+embedding
                                           # pipeline ไม่ใช่วัด EAR blink detection
        "liveness_pass":   True,
        "face_image":      images[0],
        "face_images":     images,
        "ear_samples":     [0.25, 0.24, 0.26, 0.25],
    }
    headers = {
        "Content-Type":  "application/json",
        "X-CSRF-Token":  entry["csrf_token"],
    }

    t0 = time.perf_counter()
    try:
        resp = s.post(f"{base_url}/api/checkin", json=payload, headers=headers, timeout=timeout)
        elapsed = time.perf_counter() - t0
        try:
            body = resp.json()
        except ValueError:
            body = {}
        return {
            "email": email, "status": resp.status_code, "elapsed": elapsed,
            "body": body, "error": None,
            "t_start": t0, "t_end": t0 + elapsed,
        }
    except requests.Timeout:
        elapsed = time.perf_counter() - t0
        return {"email": email, "status": None, "elapsed": elapsed, "body": {}, "error": "timeout",
                "t_start": t0, "t_end": t0 + elapsed}
    except requests.RequestException as e:
        elapsed = time.perf_counter() - t0
        return {"email": email, "status": None, "elapsed": elapsed, "body": {}, "error": f"network_error: {e}",
                "t_start": t0, "t_end": t0 + elapsed}


def _classify(r: dict) -> str:
    if r["error"] == "timeout":
        return "timeout"
    if r["error"]:
        return "network_error"
    status = r["status"]
    body = r["body"] or {}
    if status == 429:
        return "rate_limited"
    if status and status >= 500:
        return "server_error"
    if status == 403:
        return "forbidden"          # device/integrity/enrollment reject
    if status == 404:
        return "session_not_found"
    if status == 400 and body.get("spoof"):
        return "spoof_reject"       # ตัดก่อนถึง DeepFace — ถ้าเจอเยอะ แปลว่าภาพเทสมีปัญหา
    if status == 400 and body.get("already_checked"):
        return "already_checked"
    if status == 400 and "ไม่ตรง" in (body.get("error") or ""):
        return "no_match"           # ผ่าน pipeline เต็ม รวม DeepFace แล้ว — ผลลัพธ์ที่ "คาดหวัง" จาก fake embeddings
    if status == 200 and body.get("ok"):
        return "success"            # แทบเป็นไปไม่ได้ด้วย random embedding แต่เก็บไว้เผื่อ
    return f"other_{status}"


def phase_checkin(base_url: str, session_id: str, images: list[str], label: str):
    state = _load_state()
    if not state:
        print(f"ERROR: ไม่พบ session state ({STATE_FILE}) — รัน 'login' ก่อน หรือรันแบบ 'all'", file=sys.stderr)
        sys.exit(1)

    emails = [STUDENT_EMAIL_FMT.format(i) for i in range(1, STUDENT_COUNT + 1)]
    ready  = [e for e in emails if e in state]
    missing = [e for e in emails if e not in state]
    if missing:
        print(f"WARNING: {len(missing)} คนไม่มี session (login ไม่สำเร็จตอนเฟส A) — ยิงแค่ {len(ready)} คน: {missing}")

    print(f"\n=== เฟส B: ยิง /api/checkin พร้อมกัน {len(ready)} request ===\n")

    results = []
    burst_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(ready)) as ex:
        futures = {
            ex.submit(_checkin_one, base_url, email, state[email], session_id, images): email
            for email in ready
        }
        for fut in as_completed(futures):
            results.append(fut.result())
    burst_wall_time = time.perf_counter() - burst_start

    _report(results, burst_wall_time, label)


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, int(round(pct / 100.0 * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def _report(results: list[dict], burst_wall_time: float, label: str):
    from collections import Counter

    durations = sorted(r["elapsed"] for r in results)
    categories = Counter(_classify(r) for r in results)

    print("=== ผลลัพธ์ ===")
    print(f"Total requests:     {len(results)}")
    print(f"Wall time (ยิงตัวแรก → response ตัวสุดท้าย): {burst_wall_time:.2f}s")
    if durations:
        print(f"Per-request time:   min={min(durations):.3f}s  "
              f"p50={_percentile(durations,50):.3f}s  "
              f"p95={_percentile(durations,95):.3f}s  "
              f"p99={_percentile(durations,99):.3f}s  "
              f"max={max(durations):.3f}s")
    print("\nแยกตามประเภทผลลัพธ์:")
    for cat, count in categories.most_common():
        print(f"  {cat:20s} {count}")

    if categories.get("spoof_reject", 0) > 0:
        print(
            f"\n*** {categories['spoof_reject']} request โดน spoof_reject — เช็คภาพที่ใช้เทสอีกที "
            "(ต้องเป็นภาพถ่ายจริงจากกล้อง, เฟรมใน face_images ต้องแตกต่างกันจริง ไม่ใช่ไฟล์เดียวกันซ้ำ) ***"
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(RESULTS_DIR, f"{label}_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "label": label,
            "timestamp_utc": ts,
            "total_requests": len(results),
            "burst_wall_time_sec": burst_wall_time,
            "percentiles_sec": {
                "min": min(durations) if durations else None,
                "p50": _percentile(durations, 50),
                "p95": _percentile(durations, 95),
                "p99": _percentile(durations, 99),
                "max": max(durations) if durations else None,
            },
            "categories": dict(categories),
            "raw_results": [
                {k: v for k, v in r.items() if k not in ("t_start", "t_end")}
                for r in results
            ],
        }, f, ensure_ascii=False, indent=2)
    print(f"\nบันทึกผลไว้ที่ {out_path}")


# ============================================================
# Compare mode — เทียบผล 2 (หรือมากกว่า) รอบ
# ============================================================

def phase_compare(paths: list[str]):
    expanded = []
    for p in paths:
        expanded.extend(sorted(globmod.glob(p)) or [p])

    rows = []
    for p in expanded:
        if not os.path.exists(p):
            print(f"WARNING: ไม่พบไฟล์ {p} — ข้าม")
            continue
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        rows.append(d)

    if not rows:
        print("ไม่มีผลลัพธ์ให้เทียบ")
        return

    print(f"{'label':20s} {'total':>6s} {'success':>8s} {'no_match':>9s} {'spoof':>6s} "
          f"{'429':>5s} {'p50':>7s} {'p95':>7s} {'p99':>7s} {'wall':>8s}")
    for d in rows:
        cat = d.get("categories", {})
        pct = d.get("percentiles_sec", {})
        print(
            f"{d.get('label','?'):20s} {d.get('total_requests',0):>6d} "
            f"{cat.get('success',0):>8d} {cat.get('no_match',0):>9d} {cat.get('spoof_reject',0):>6d} "
            f"{cat.get('rate_limited',0):>5d} "
            f"{pct.get('p50',0) or 0:>7.3f} {pct.get('p95',0) or 0:>7.3f} {pct.get('p99',0) or 0:>7.3f} "
            f"{d.get('burst_wall_time_sec',0):>8.2f}"
        )


# ============================================================
# Warmup mode — ยิง /api/checkin ทีละ request (sequential) ให้ทุก gunicorn
# worker โหลด TF/DeepFace/Fasnet เข้า memory ก่อนวัด burst 50 คนจริง
#
# ข้อจำกัดที่ต้องรู้ (อ่านก่อนเชื่อผลลัพธ์): request ยิงทีละตัวเรียงกัน ไม่ได้
# รับประกันว่าจะกระจายไปครบทุก worker — gunicorn sync worker แข่งกันด้วย
# accept-lock ก่อนรับ request แต่ละครั้ง worker ที่ warm แล้ว (ตอบเร็วกว่า) มี
# แนวโน้มกลับไปรอ accept-lock รอบใหม่ได้เร็วกว่า worker ที่ยังโหลดโมเดลอยู่
# (ช้ากว่า) จึงมีโอกาสชนะ lock ซ้ำ ๆ ทำให้ worker เย็นบางตัวอาจไม่เคยถูกเรียก
# เลยระหว่าง warmup — ตัวเลข elapsed/variance ด้านล่างเป็นแค่สัญญาณเสริม
# (proxy) ไม่ใช่ ground truth ต้องเช็ค RAM บน Railway metrics tab คู่กันเสมอ
# (ดู docs/review/09-load-test-prep.md §3)
# ============================================================

def phase_warmup(base_url: str, session_id: str, images: list[str], count: int):
    email = STUDENT_EMAIL_FMT.format(1)
    print(f"=== Warmup: {count} sequential /api/checkin request ({email}) ===\n")
    print(
        "หมายเหตุ: request เรียงลำดับทีละตัว ไม่รับประกันว่าจะกระจายไปครบทุก worker —\n"
        "ดู RAM บน Railway metrics tab (~906MB x จำนวน worker) เป็นตัวยืนยันจริง ไม่ใช่แค่ elapsed ด้านล่าง\n"
    )

    entry = _login_one(base_url, email)
    if not entry:
        print(f"ERROR: login {email} ไม่สำเร็จ — เช็คว่ารัน scripts/seed_load_test.py แล้วหรือยัง", file=sys.stderr)
        sys.exit(1)

    elapsed_list = []
    for i in range(1, count + 1):
        result = _checkin_one(base_url, email, entry, session_id, images, timeout=WARMUP_TIMEOUT_SEC)
        elapsed_list.append(result["elapsed"])
        status_note = f"status={result['status']}" if result["status"] is not None else f"error={result['error']}"
        print(f"[{i:2d}/{count}] elapsed={result['elapsed']:.3f}s  {status_note}")

    print("\n=== Warmup summary ===")
    print("elapsed: " + ", ".join(f"{e:.3f}s" for e in elapsed_list))

    if len(elapsed_list) >= 3:
        last3 = elapsed_list[-3:]
        if max(last3) > 2 * min(last3):
            print(
                f"\n*** WARNING: elapsed ของ 3 request สุดท้ายต่างกันเกิน 2 เท่า "
                f"({min(last3):.3f}s - {max(last3):.3f}s) — อาจยังไม่ warm ครบทุก worker "
                "เพิ่ม --count หรือเช็ค RAM บน Railway metrics tab ก่อนยิง burst จริง ***"
            )
        else:
            print(
                f"\nlast 3 elapsed ใกล้เคียงกัน ({min(last3):.3f}s - {max(last3):.3f}s) — "
                "สัญญาณดี แต่ต้องเช็ค RAM plateau บน Railway metrics tab เพื่อยืนยันอีกที "
                "ก่อนเชื่อว่า warm ครบทุก worker จริง (ดูคำเตือนบนสุดของฟังก์ชันนี้)"
            )
    else:
        print("\n(count < 3 — ข้ามการเช็ค variance)")


# ============================================================
# Smoke mode — เทสเดี่ยว 1 คน 1 request พร้อมวินิจฉัยว่าตกที่ขั้นไหนของ pipeline
# ============================================================

def _diagnose_smoke(result: dict):
    """วินิจฉัยผลลัพธ์ 1 request จาก /api/checkin — match ตาม error message จริง
    ใน app/routes/api_checkin.py (ไม่ใช่ข้อความที่จำมา)"""
    print("\n=== วินิจฉัย ===")

    if result["error"] == "timeout":
        print("Timeout — server ไม่ตอบภายในเวลาที่กำหนด เช็ค --url ว่าถูกต้องและ Railway deploy กำลังรันอยู่จริง")
        return
    if result["error"]:
        print(f"Network error: {result['error']} — เช็ค --url ว่าถูกต้องและเข้าถึงได้จากเครื่องนี้")
        return

    status = result["status"]
    body = result["body"] or {}
    err = body.get("error") or ""

    if status == 429:
        print("429 Rate limited — /api/checkin จำกัด 5 ต่อนาทีต่อ user (key=user:{uid}) รอสักครู่แล้วลองใหม่")
        return

    if status == 503 and "ขัดข้องชั่วคราว" in err:
        print(
            "503 anti-spoof system unavailable — Fasnet (หรือทุก voting layer) พังจนรันไม่ได้ "
            "**ไม่ใช่การตัดสินว่าเป็น spoof** (F-16, docs/review/10-moire-frr-investigation.md §16)\n"
            "*** อย่าเปลี่ยนภาพเทส — เช็ค server log หา '[FASNET] inference error' / "
            "'[COMBINED_SPOOF] ... failing CLOSED' เพื่อดู root cause จริงแทน ***"
        )
        return

    if status is not None and status >= 500:
        print(f"{status} Server error — ดู Railway logs ของ service (Deployments > Logs) เพื่ออ่าน stack trace จริง")
        return

    if status == 403 and err == "ข้อมูลชีวมาตรไม่สมบูรณ์ — กรุณาลงทะเบียนใบหน้าใหม่อีกครั้ง":
        print(
            "403 integrity hash ไม่ผ่าน (verify_embedding_integrity, api_checkin.py) — เกิดก่อนถึง extract_embedding() เสมอ\n"
            "แปลว่า EMBEDDING_INTEGRITY_SALT บน Railway ไม่ตรงกับ salt ที่ใช้ตอนรัน seed_load_test.py\n"
            "*** timing ที่วัดได้ตอนนี้ไม่ครอบคลุม DeepFace เลย — ต้องแก้ salt ให้ตรงกันก่อน ***\n"
            "ไม่งั้น load test 50 คนจะวัด compute cost ต่ำกว่าความจริงมาก เพราะทุก request ตกก่อนถึงส่วนที่หนักที่สุด"
        )
        return

    if status == 403 and err == "คุณไม่ได้ลงทะเบียนในรายวิชานี้":
        print("403 ไม่ได้ลงทะเบียนเรียน — seed_load_test.py อาจไม่ได้ใส่ course_enrollments ให้ครบ หรือ --session-id ผิด course ตรวจ seed script")
        return

    if status == 403 and err == "อุปกรณ์นี้ถูกผูกกับบัญชีอื่นแล้ว":
        print("403 device_id ชนกับ user อื่น — smoke test ไม่ได้ส่ง X-Device-ID เอง ไม่ควรเกิดกรณีนี้ เช็คว่า user นี้มี device_id ค้างจาก DB")
        return

    if status == 404 and err == "ไม่พบ session":
        print("404 ไม่พบ session — session_id ผิด หรือ session ถูกลบไปแล้ว/ยังไม่ได้ seed")
        return

    if status == 400 and body.get("spoof"):
        if err == "ตรวจพบหน้าจอมือถือ — กรุณาใช้ใบหน้าจริงเท่านั้น":
            print("400 spoof reject — ตกที่ Moiré FFT (คิดว่าเป็นภาพถ่ายจากหน้าจอ) — ต้องเป็นภาพถ่ายจริงจากกล้อง ไม่ใช่จากจอ/สิ่งพิมพ์")
        elif err == "ตรวจพบภาพจากหน้าจอ — กรุณาใช้ใบหน้าจริงเท่านั้น":
            print("400 spoof reject — ตกที่ Screen-Texture FFT (เจอ spectral peak ของจอ) — ต้องเป็นภาพถ่ายจริงจากกล้อง ไม่ใช่จากจอ/สิ่งพิมพ์")
        elif err == "ตรวจพบภาพนิ่ง — กรุณาใช้ใบหน้าจริงเท่านั้น":
            print(
                "400 spoof reject — ตกที่ Temporal-Variance (เฟรมใน face_images เหมือนกันหมด = ภาพนิ่ง)\n"
                "เช็คว่า --images ที่ส่งเข้ามาเป็นไฟล์ต่างกันจริง (burst shot คนละเฟรม) ไม่ใช่ไฟล์เดียวกัน copy ซ้ำ"
            )
        elif err == "ตรวจพบรูปถ่ายหรือหน้าจอ — กรุณาใช้ใบหน้าจริงเท่านั้น":
            print("400 spoof reject — ตกที่ MiniFASNet anti-spoof model — ต้องเป็นภาพถ่ายจริงจากกล้อง")
        else:
            print(f"400 spoof reject — error='{err}' ไม่ match error string ที่รู้จัก เช็คว่า api_checkin.py เปลี่ยนข้อความไปหรือยัง")
        return

    if status == 400 and err == "รูปภาพไม่ถูกต้อง — กรุณาถ่ายใหม่อีกครั้ง":
        print("400 frame validation fail (server_validate_frame) — เช็ค JPEG format / ขนาดไฟล์ 3-500KB / ความละเอียด 160x120-1920x1080 / blur variance >= 8 / color std-dev >= 2.0")
        return

    if status == 400 and err == "ข้อมูลภาพต่อเนื่องไม่ครบ กรุณาลองใหม่":
        print("400 face_images ไม่ครบ >=2 เฟรม — เช็คว่า --images ส่งอย่างน้อย 2 ไฟล์เข้ามาจริง")
        return

    if status == 400 and err == "ยังไม่ได้ลงทะเบียนใบหน้า":
        print("400 ยังไม่มี biometrics — seed_load_test.py ไม่ได้ใส่ face_embeddings ให้ user นี้ ตรวจ seed script")
        return

    if status == 400 and "ไม่ตรง" in err:
        print(
            "400 ใบหน้าไม่ตรง — ผ่าน pipeline เต็มรวม extract_embedding() (DeepFace) แล้ว\n"
            "*** นี่คือผลลัพธ์ที่ถูกต้อง เพราะ seed embedding เป็นเลขสุ่ม — timing ที่วัดได้สมจริง ใช้เป็น baseline ได้เลย ***"
        )
        return

    if status == 400 and body.get("already_checked"):
        print("400 เช็คชื่อไปแล้ว — session/student นี้มี attendance record ค้างอยู่ ลบ record หรือเปลี่ยน session แล้วลองใหม่")
        return

    if status == 200 and body.get("ok"):
        print("200 สำเร็จ — ไม่คาดคิดกับ seed embedding แบบสุ่ม แต่ pipeline ผ่านครบทุกขั้นรวม DeepFace แล้ว timing ใช้เป็น baseline ได้")
        return

    print(f"ไม่ match เงื่อนไขที่รู้จักในสคริปต์นี้ — status={status} body={body} — อ่าน api_checkin.py เพิ่มเติมเพื่อหาสาเหตุ")


def phase_smoke(base_url: str, session_id: str, images: list[str]):
    email = STUDENT_EMAIL_FMT.format(1)
    print(f"=== Smoke test: login {email} + ยิง /api/checkin 1 ครั้ง ===\n")

    entry = _login_one(base_url, email)
    if not entry:
        print(f"ERROR: login {email} ไม่สำเร็จ — เช็คว่ารัน scripts/seed_load_test.py แล้วหรือยัง", file=sys.stderr)
        sys.exit(1)

    result = _checkin_one(base_url, email, entry, session_id, images)

    print(f"\nHTTP status: {result['status']}")
    print(f"Elapsed:     {result['elapsed']:.3f}s")
    print("Response body:")
    print(json.dumps(result["body"], ensure_ascii=False, indent=2))
    if result["error"]:
        print(f"Error: {result['error']}")

    _diagnose_smoke(result)


# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Load test /api/checkin (SmartCheck)")
    parser.add_argument("mode", choices=["login", "checkin", "all", "compare", "smoke", "warmup"])
    parser.add_argument("--url", help="Base URL ของ server (Railway deployment เท่านั้น เช่น https://xxxx.up.railway.app)")
    parser.add_argument("--images", nargs="+", help="path ไฟล์ภาพ JPEG จริง >=2 ไฟล์ (คนละเฟรม ห้ามซ้ำกัน)")
    parser.add_argument("--session-id", help="session id ที่เปิดอยู่ — ไม่ระบุจะหาอัตโนมัติจาก Supabase")
    parser.add_argument("--label", default="run", help="ชื่อ label สำหรับไฟล์ผลลัพธ์ เช่น 1worker / 4workers")
    parser.add_argument("--workers", type=int, default=1,
                         help="mode=warmup: จำนวน gunicorn worker ของรอบที่กำลังเทส — ใช้คำนวณ default ของ --count (=3xworkers)")
    parser.add_argument("--count", type=int,
                         help="mode=warmup: จำนวน sequential request ที่จะยิง (default = 3 x --workers)")
    parser.add_argument("compare_files", nargs="*", help="(เฉพาะ mode=compare) path ผลลัพธ์ .json ที่จะเทียบ")
    args = parser.parse_args()

    if args.mode == "compare":
        files = args.compare_files or sys.argv[2:]
        if not files:
            print("ERROR: ระบุไฟล์ผลลัพธ์ที่จะเทียบ เช่น: python scripts/load_test.py compare results/1worker_*.json results/4workers_*.json", file=sys.stderr)
            sys.exit(1)
        phase_compare(files)
        return

    if not args.url:
        print("ERROR: ต้องระบุ --url (Railway deployment เท่านั้น — gunicorn รันบน Windows local ไม่ได้)", file=sys.stderr)
        sys.exit(1)
    base_url = args.url.rstrip("/")

    if args.mode == "smoke":
        if not args.images:
            print("ERROR: ต้องระบุ --images ไฟล์ภาพ JPEG จริงอย่างน้อย 2 ไฟล์ (ดู docstring บนสุดของไฟล์นี้)", file=sys.stderr)
            sys.exit(1)
        images = _load_images(args.images)
        session_id = args.session_id or _discover_session_id()
        phase_smoke(base_url, session_id, images)
        return

    if args.mode == "warmup":
        if not args.images:
            print("ERROR: ต้องระบุ --images ไฟล์ภาพ JPEG จริงอย่างน้อย 2 ไฟล์ (ดู docstring บนสุดของไฟล์นี้)", file=sys.stderr)
            sys.exit(1)
        images = _load_images(args.images)
        session_id = args.session_id or _discover_session_id()
        count = args.count if args.count is not None else 3 * args.workers
        phase_warmup(base_url, session_id, images, count)
        return

    if args.mode in ("login", "all"):
        phase_login(base_url)

    if args.mode in ("checkin", "all"):
        if not args.images:
            print("ERROR: ต้องระบุ --images ไฟล์ภาพ JPEG จริงอย่างน้อย 2 ไฟล์ (ดู docstring บนสุดของไฟล์นี้)", file=sys.stderr)
            sys.exit(1)
        images = _load_images(args.images)
        session_id = args.session_id or _discover_session_id()
        phase_checkin(base_url, session_id, images, args.label)


if __name__ == "__main__":
    main()
