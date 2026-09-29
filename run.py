from pathlib import Path

from app import create_app
from app.config import _IS_PRODUCTION

app = create_app()

if __name__ == "__main__":
    import logging
    from logging.handlers import RotatingFileHandler

    checkin_log = RotatingFileHandler(
        Path(__file__).resolve().parent / "checkin-diagnostics.log",
        maxBytes=2_000_000, backupCount=2, encoding="utf-8",
    )
    checkin_log.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger("smartcheck.checkin").addHandler(checkin_log)
    # Same switch as config.py: FLASK_DEBUG=0 or FLASK_ENV=production turns the
    # Werkzeug debugger and reloader off.
    app.run(debug=not _IS_PRODUCTION, port=5000)
