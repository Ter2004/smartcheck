"""
Migration: FaceNet128 → FaceNet512
====================================
Run this script ONCE after deploying the Facenet512 refactor.

What it does:
  1. Clears face_embeddings for all students (128-D incompatible with 512-D)
  2. Sets consent_given = False (forces re-enrollment)
  3. Clears face_centroid if the column still exists (no longer used)
  4. Prints a summary of affected rows

After running this script, ALL students must re-enroll.
"""

import os
import sys

# Load .env before anything else
from dotenv import load_dotenv
load_dotenv()

from supabase import create_client  # noqa: E402

SUPABASE_URL         = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    print("ERROR: SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env")
    sys.exit(1)

db = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

print("=== SmartCheck: FaceNet512 Migration ===\n")

# ─── 1. Report current state ──────────────────────────────────────────────────
rows = db.table("student_biometrics").select("user_id, consent_given").execute().data or []
enrolled = [r for r in rows if r.get("consent_given")]
print(f"Total students in biometrics table : {len(rows)}")
print(f"Currently enrolled (consent_given) : {len(enrolled)}")
print()

if not rows:
    print("No rows found — nothing to migrate.")
    sys.exit(0)

confirm = input("This will CLEAR all face embeddings and force re-enrollment.\nType 'yes' to continue: ").strip()
if confirm.lower() != "yes":
    print("Migration cancelled.")
    sys.exit(0)

# ─── 2. Clear embeddings and reset consent_given ─────────────────────────────
update_payload = {"face_embeddings": None, "consent_given": False}

# Use a filter that matches all rows (user_id is never this value)
DUMMY_FILTER = ("user_id", "neq", "00000000-0000-0000-0000-000000000000")

try:
    db.table("student_biometrics") \
        .update({**update_payload, "face_centroid": None}) \
        .neq(*DUMMY_FILTER[1:]) \
        .execute()
    print("✓ Cleared face_embeddings, face_centroid, reset consent_given.")
except Exception:
    # face_centroid column may not exist
    db.table("student_biometrics") \
        .update(update_payload) \
        .neq("user_id", "00000000-0000-0000-0000-000000000000") \
        .execute()
    print("✓ Cleared face_embeddings, reset consent_given.")
    print("  (face_centroid column not found — skipped)")

print()
print("=== Migration complete ===")
print(f"  {len(rows)} student(s) must re-enroll with the new FaceNet512 model.")
print()
print("Optional: drop the unused face_centroid column via Supabase SQL editor:")
print("  ALTER TABLE student_biometrics DROP COLUMN IF EXISTS face_centroid;")
