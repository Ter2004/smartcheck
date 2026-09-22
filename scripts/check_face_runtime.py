"""Exercise the actual local face models without writing student/database data."""
import base64
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    logging.basicConfig(level=logging.INFO)
    print(f"Python: {sys.executable}", flush=True)
    import tensorflow as tf
    import cv2
    from deepface import DeepFace

    print(f"TensorFlow: {tf.__version__}", flush=True)
    sample = ROOT / "test_images" / "real_01_sharp_face1.jpg"
    image = cv2.imread(str(sample))
    if image is None:
        raise RuntimeError(f"Cannot read test image: {sample}")
    faces = DeepFace.extract_faces(
        img_path=image, detector_backend="opencv",
        anti_spoofing=True, enforce_detection=True,
    )
    if not faces or any("antispoof_score" not in face for face in faces):
        raise RuntimeError("Fasnet did not return anti-spoof scores")
    print("PASS: Fasnet inference", flush=True)
    vectors = DeepFace.represent(
        img_path=image, model_name="Facenet512",
        detector_backend="opencv", enforce_detection=True,
    )
    if not vectors or len(vectors[0]["embedding"]) != 512:
        raise RuntimeError("FaceNet512 did not return a 512-dimensional vector")
    print("PASS: FaceNet512 inference", flush=True)
    from app.services.face_service import spoof_check_with_embedding
    result = spoof_check_with_embedding(base64.b64encode(sample.read_bytes()).decode("ascii"))
    if result.get("system_failure") or result.get("layers", {}).get("fasnet", {}).get("spoof_score") is None:
        raise RuntimeError("Application anti-spoof pipeline unavailable; inspect preceding logs")
    # A spoof rejection is a valid model decision, not a runtime failure.
    if result["is_real"] and not result.get("embedding"):
        raise RuntimeError("Application pipeline returned no embedding")
    print("PASS: application anti-spoof pipeline", flush=True)
    print("PASS: local test image processed. Live camera/API/database are not covered.")


if __name__ == "__main__":
    main()
