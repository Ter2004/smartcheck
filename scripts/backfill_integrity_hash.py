"""
scripts/backfill_integrity_hash.py
===================================
คำนวณ integrity_hash สำหรับ rows ใน student_biometrics ที่ยังไม่มี hash
รัน 1 ครั้งหลัง deploy (หลังจาก A4 เปลี่ยน verify_embedding_integrity ให้ fail-close)

Usage:
    FLASK_SECRET_KEY=... EMBEDDING_INTEGRITY_SALT=... \\
    SUPABASE_URL=... SUPABASE_SERVICE_KEY=... \\
    python scripts/backfill_integrity_hash.py [--dry-run]

Options:
    --dry-run   แสดงจำนวน rows ที่จะ update โดยไม่ write จริง
"""
import sys
import os

# ── เพิ่ม project root ลง sys.path ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

DRY_RUN = "--dry-run" in sys.argv


def main():
    from supabase import create_client
    from app.services.security_service import compute_embedding_integrity_hash

    supabase_url = os.getenv("SUPABASE_URL")
    service_key  = os.getenv("SUPABASE_SERVICE_KEY")
    salt         = os.getenv("EMBEDDING_INTEGRITY_SALT")

    if not all([supabase_url, service_key, salt]):
        print("ERROR: SUPABASE_URL, SUPABASE_SERVICE_KEY, EMBEDDING_INTEGRITY_SALT must be set",
              file=sys.stderr)
        sys.exit(1)

    client = create_client(supabase_url, service_key)

    # ดึงเฉพาะ rows ที่มี face_embeddings แต่ยังไม่มี integrity_hash
    res = (
        client.table("student_biometrics")
        .select("user_id, face_embeddings")
        .not_.is_("face_embeddings", "null")
        .is_("integrity_hash", "null")
        .execute()
    )
    rows = res.data or []
    print(f"Found {len(rows)} rows missing integrity_hash")

    if DRY_RUN:
        print("DRY RUN — no changes written")
        return

    updated = 0
    errors  = 0
    for row in rows:
        user_id    = row["user_id"]
        embeddings = row["face_embeddings"]
        if not embeddings:
            continue
        try:
            h = compute_embedding_integrity_hash(user_id, embeddings, salt)
            client.table("student_biometrics") \
                .update({"integrity_hash": h}) \
                .eq("user_id", user_id) \
                .execute()
            updated += 1
            print(f"  ✓ {user_id[:8]}...")
        except Exception as e:
            errors += 1
            print(f"  ✗ {user_id[:8]}... ERROR: {e}", file=sys.stderr)

    print(f"\nDone: {updated} updated, {errors} errors")


if __name__ == "__main__":
    main()
