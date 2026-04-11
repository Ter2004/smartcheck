from app import supabase_admin


def get_user_by_id(user_id: str) -> dict | None:
    """ดึงข้อมูล user จาก id"""
    res = supabase_admin.table("users").select("*").eq("id", user_id).maybe_single().execute()
    return res.data


def get_user_by_email(email: str) -> dict | None:
    """ดึงข้อมูล user จาก email"""
    res = supabase_admin.table("users").select("*").eq("email", email).maybe_single().execute()
    return res.data
