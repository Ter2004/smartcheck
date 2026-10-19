"""Exercise the actual local face models (v2) without writing student/database data."""
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    logging.basicConfig(level=logging.INFO)
    print(f"Python: {sys.executable}", flush=True)
    import cv2
    import insightface
    import onnxruntime
    from app.runtime_check import check_face_runtime
    from app.services import face_engine, fasnet_onnx

    check_face_runtime()
    print(f"insightface {insightface.__version__}, onnxruntime {onnxruntime.__version__}", flush=True)
    print(f"warm-up {face_engine.warm_up():.2f}s", flush=True)
    image = cv2.imread(str(ROOT / "test_images" / "real_01_sharp_face1.jpg"))
    if image is None:
        raise RuntimeError("Cannot read test image")
    face = face_engine.analyze(image, clahe=False)
    norm = float((face["embedding"] ** 2).sum() ** 0.5)
    if abs(norm - 1.0) > 1e-3:
        raise RuntimeError(f"embedding norm {norm}")
    print(f"PASS: ArcFace embedding (512-D, norm {norm:.4f}), yaw {face['yaw_deg']:.1f} deg", flush=True)
    is_real, score = fasnet_onnx.analyze(image, face_engine.box_xywh(face, image.shape))
    print(f"PASS: Fasnet ONNX inference (real={is_real}, score={score:.3f})", flush=True)


if __name__ == "__main__":
    main()
