import importlib.util
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_BUFFALO_L = ("det_10g.onnx", "w600k_r50.onnx", "1k3d68.onnx")
_FASNET = ("fasnet_minifasnetv2_2.7_80x80.onnx", "fasnet_minifasnetv1se_4.0_80x80.onnx", "antispoof.onnx")


def check_face_runtime():
    """Catch a missing face runtime or model file before accepting camera requests."""
    missing = [m for m in ("onnxruntime", "insightface") if importlib.util.find_spec(m) is None]
    models = Path.home() / ".insightface" / "models" / "buffalo_l"
    missing += [f"{models / f}" for f in _BUFFALO_L if not (models / f).exists()]
    local = _PROJECT_ROOT / "app" / "services" / "models"
    missing += [f"{local / f}" for f in _FASNET if not (local / f).exists()]
    if missing:
        python = _PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
        raise SystemExit(
            "SmartCheck cannot start: the face runtime is incomplete.\n"
            f"Missing: {', '.join(missing)}\n"
            f"Current Python: {sys.executable}\n"
            f'Install with "{python}" -m pip install -r requirements.txt; buffalo_l downloads to '
            "~/.insightface/models on first FaceAnalysis(name='buffalo_l') (see Dockerfile)."
        )
