import importlib.util
import sys
from pathlib import Path


def check_face_runtime():
    """Catch an incomplete TensorFlow install before accepting camera requests."""
    spec = importlib.util.find_spec("tensorflow")
    if spec is None or spec.origin is None:
        python = Path(__file__).resolve().parent / "venv" / "Scripts" / "python.exe"
        raise SystemExit(
            "SmartCheck cannot start: TensorFlow is missing or incomplete.\n"
            f"Current Python: {sys.executable}\n"
            "Run start-smartcheck.cmd from the outer project folder, or use:\n"
            f'"{python}" "{Path(__file__).resolve()}"'
        )


if __name__ == "__main__":
    check_face_runtime()

from app import create_app

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
    app.run(debug=True, port=5000)
