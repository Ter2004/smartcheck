import secrets
from functools import wraps
from flask import Blueprint, render_template, request, redirect, url_for, session, flash, current_app
from app import supabase, supabase_admin
from app import limiter as _limiter
from flask_limiter.util import get_remote_address
from app.models.user_model import get_user_by_id

auth_bp = Blueprint("auth", __name__)


# ============================================================
# DECORATORS — ใช้ครอบ route เพื่อเช็คสิทธิ์
# ============================================================

def login_required(f):
    """ต้อง login ก่อนถึงจะเข้าหน้านี้ได้"""
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("กรุณาเข้าสู่ระบบก่อน", "warning")
            return redirect(url_for("auth.login"))
        return f(*args, **kwargs)
    return decorated


def role_required(*roles):
    """เช็คว่า role ตรงกับที่กำหนดหรือไม่ เช่น @role_required('admin', 'teacher')"""
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if "user_role" not in session or session["user_role"] not in roles:
                flash("คุณไม่มีสิทธิ์เข้าถึงหน้านี้", "danger")
                return redirect(url_for("auth.login"))
            return f(*args, **kwargs)
        return decorated
    return decorator


# ============================================================
# ROUTES
# ============================================================

@auth_bp.route("/")
def index():
    """หน้าแรก — redirect ตาม role"""
    if "user_id" not in session:
        return redirect(url_for("auth.login"))
    return _redirect_by_role(session.get("user_role"))


@auth_bp.route("/login", methods=["GET", "POST"])
@_limiter.limit("10 per minute", methods=["POST"], key_func=get_remote_address)
def login():
    if request.method == "GET":
        return render_template("auth/login.html")

    email = request.form.get("email", "").strip()
    password = request.form.get("password", "")

    try:
        # 1) Supabase Auth — sign in
        res = supabase.auth.sign_in_with_password({"email": email, "password": password})
        sb_user = res.user

        # 2) ดึง role จากตาราง users
        user = get_user_by_id(sb_user.id)
        if not user:
            flash("ไม่พบบัญชีในระบบ กรุณาติดต่อผู้ดูแล", "danger")
            return redirect(url_for("auth.login"))

        if not user.get("is_active"):
            flash("บัญชีถูกปิดการใช้งาน กรุณาติดต่อผู้ดูแล", "danger")
            return redirect(url_for("auth.login"))

        # 3) Session regeneration — ป้องกัน Session Fixation Attack
        # ต้องทำก่อน set ข้อมูล user ใด ๆ ลง session
        #
        # flask-session filesystem backend: session.clear() เพียงอย่างเดียว
        # ไม่เปลี่ยน session ID — มันแค่ overwrite ไฟล์เดิมด้วย dict ว่าง
        # ต้อง assign session.sid ใหม่เพื่อบังคับให้สร้างไฟล์ใหม่ + cookie ใหม่
        _old_sid = getattr(session, "sid", None)
        session.clear()

        # เปลี่ยน sid → flask-session จะ save ลงไฟล์ใหม่และ set cookie ใหม่
        if hasattr(session, "sid"):
            session.sid = secrets.token_hex(32)

        # ลบ session file เก่าออกจาก filesystem (best-effort)
        if _old_sid and hasattr(current_app.session_interface, "cache"):
            try:
                _prefix = getattr(current_app.session_interface, "key_prefix", "session:")
                current_app.session_interface.cache.delete(_prefix + _old_sid)
            except Exception:
                pass

        # 4) เก็บ session ใหม่หลัง regenerate แล้ว
        session["user_id"]      = str(sb_user.id)
        session["user_role"]    = user["role"]
        session["user_name"]    = user["full_name"]
        session["access_token"] = res.session.access_token
        # Sprint 1C: per-session CSRF token (สร้างใหม่พร้อม session ใหม่)
        session["csrf_token"]   = secrets.token_hex(32)
        session.modified        = True

        return _redirect_by_role(user["role"])

    except Exception as e:
        flash("อีเมลหรือรหัสผ่านไม่ถูกต้อง", "danger")
        return redirect(url_for("auth.login"))


@auth_bp.route("/register")
def register():
    """ปิดการสมัครสมาชิกด้วยตัวเอง — Admin เป็นผู้สร้าง account ผ่าน CSV import เท่านั้น"""
    flash("การสมัครสมาชิกถูกปิด — กรุณาติดต่อผู้ดูแลระบบ", "warning")
    return redirect(url_for("auth.login"))


@auth_bp.route("/logout")
def logout():
    try:
        supabase.auth.sign_out()
    except Exception:
        pass
    session.clear()
    flash("ออกจากระบบแล้ว", "info")
    return redirect(url_for("auth.login"))


# ============================================================
# HELPERS
# ============================================================

def _redirect_by_role(role: str):
    """Redirect ไปหน้า dashboard ตาม role"""
    if role == "admin":
        return redirect(url_for("admin.dashboard"))
    elif role == "teacher":
        return redirect(url_for("teacher.dashboard"))
    else:
        return redirect(url_for("student.dashboard"))
