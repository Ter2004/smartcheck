import importlib.util
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def check_face_runtime():
    """Catch an incomplete TensorFlow install before accepting camera requests."""
    spec = importlib.util.find_spec("tensorflow")
    if spec is None or spec.origin is None:
        python = _PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
        raise SystemExit(
            "SmartCheck cannot start: TensorFlow is missing or incomplete.\n"
            f"Current Python: {sys.executable}\n"
            "Run start-smartcheck.cmd from the outer project folder, or use:\n"
            f'"{python}" "{_PROJECT_ROOT / "run.py"}"'
        )
