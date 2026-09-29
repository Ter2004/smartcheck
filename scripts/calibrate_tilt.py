"""Measure head-tilt (roll) detection on LFW photos rotated by known angles.

For N identities with two photos, photo 2 is rotated by each angle and run
through RetinaFace. Reports, per angle:
  roll_delta   measured roll minus photo 2's own roll (expect ≈ -angle:
               cv2's positive angle is counter-clockwise, which lifts the
               image-right eye, i.e. negative roll)
  yaw_raw      nose offset in image axes (what liveness_challenge used)
  yaw_aligned  nose offset along the eye line (rotation-invariant)
  same/other   FaceNet cosine to photo 1 of the same / next identity,
               with the crop as is and de-rotated to a level eye line.

Synthetic rotation is not a real head tilt (no perspective change); it
calibrates the geometry, not the population. Run from the project root:
  venv/Scripts/python.exe scripts/calibrate_tilt.py --people 60 --out tilt.json
"""
import argparse
import json
import math
import statistics
from pathlib import Path

import cv2
import numpy as np

ANGLES = [-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25]


def eye_points(landmarks):
    a, b = (np.asarray(landmarks[k], float)[:2] for k in ("right_eye", "left_eye"))
    return (a, b) if a[0] <= b[0] else (b, a)  # image-left, image-right


def roll_deg(landmarks):
    left, right = eye_points(landmarks)
    return math.degrees(math.atan2(right[1] - left[1], right[0] - left[0]))


def yaw_raw(landmarks):
    left, right = eye_points(landmarks)
    nose = np.asarray(landmarks["nose"], float)[:2]
    return float((nose[0] - (left[0] + right[0]) / 2) / abs(right[0] - left[0]))


def yaw_aligned(landmarks):
    left, right = eye_points(landmarks)
    nose = np.asarray(landmarks["nose"], float)[:2]
    axis = right - left
    return float(np.dot(nose - (left + right) / 2, axis) / np.dot(axis, axis))


def embed(face_service, crop):
    crop = face_service.normalize_illumination(crop)
    rep = face_service._call_deepface("represent", img_path=crop, model_name="Facenet512",
                                      detector_backend="skip")
    v = np.asarray(rep[0]["embedding"], np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


def crops(img, face):
    """(crop as is, crop de-rotated about the eye midpoint)."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in face["facial_area"])
    mx, my = int((x2 - x1) * 0.1), int((y2 - y1) * 0.1)
    plain = img[max(0, y1 - my):min(h, y2 + my), max(0, x1 - mx):min(w, x2 + mx)]
    left, right = eye_points(face["landmarks"])
    centre = tuple(((left + right) / 2).tolist())
    m = cv2.getRotationMatrix2D(centre, roll_deg(face["landmarks"]), 1.0)
    level = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    cx, cy = m @ np.array([(x1 + x2) / 2, (y1 + y2) / 2, 1.0])
    half_w, half_h = (x2 - x1) / 2 + mx, (y2 - y1) / 2 + my
    aligned = level[max(0, int(cy - half_h)):min(h, int(cy + half_h)),
                    max(0, int(cx - half_w)):min(w, int(cx + half_w))]
    return plain, aligned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lfw", type=Path, default=Path(".build/datasets/lfw/lfw"))
    ap.add_argument("--people", type=int, default=60)
    ap.add_argument("--out", type=Path, default=Path("tilt-calibration.json"))
    args = ap.parse_args()

    from retinaface import RetinaFace
    from app.services import face_service

    people = [p for p in sorted(args.lfw.iterdir()) if len(list(p.glob("*.jpg"))) >= 2][:args.people]
    detect = lambda img: RetinaFace.detect_faces(img, threshold=0.9, allow_upscaling=False)

    def main_face(img):
        faces = detect(img)
        if not isinstance(faces, dict) or not faces:
            return None
        return max(faces.values(), key=lambda f: (f["facial_area"][2] - f["facial_area"][0])
                   * (f["facial_area"][3] - f["facial_area"][1]))

    refs = []
    for person in people:
        a = cv2.imread(str(sorted(person.glob("*.jpg"))[0]))
        face = main_face(a)
        refs.append(embed(face_service, crops(a, face)[1]) if face else None)

    rows = []
    for i, person in enumerate(people):
        b = cv2.imread(str(sorted(person.glob("*.jpg"))[1]))
        base = main_face(b)
        if base is None or refs[i] is None:
            continue
        h, w = b.shape[:2]
        for angle in ANGLES:
            m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            rotated = cv2.warpAffine(b, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
            face = main_face(rotated)
            if face is None:
                rows.append({"person": person.name, "angle": angle, "detected": False})
                continue
            plain, aligned = crops(rotated, face)
            e_plain, e_aligned = embed(face_service, plain), embed(face_service, aligned)
            other = refs[(i + 1) % len(people)]
            rows.append({
                "person": person.name, "angle": angle, "detected": True,
                "roll_delta": round(roll_deg(face["landmarks"]) - roll_deg(base["landmarks"]), 2),
                "yaw_raw": round(yaw_raw(face["landmarks"]), 3),
                "yaw_aligned": round(yaw_aligned(face["landmarks"]), 3),
                "same_plain": round(float(e_plain @ refs[i]), 4),
                "same_aligned": round(float(e_aligned @ refs[i]), 4),
                "other_aligned": round(float(e_aligned @ other), 4) if other is not None else None,
            })

    summary = {}
    for angle in ANGLES:
        hit = [r for r in rows if r["angle"] == angle and r["detected"]]
        pick = lambda k: [r[k] for r in hit if r.get(k) is not None]
        summary[angle] = {
            "n": len(hit), "missed": sum(1 for r in rows if r["angle"] == angle and not r["detected"]),
            "roll_delta_median": round(statistics.median(pick("roll_delta")), 2),
            "roll_delta_min_max": [min(pick("roll_delta")), max(pick("roll_delta"))],
            "abs_yaw_raw_p90": round(sorted(map(abs, pick("yaw_raw")))[int(0.9 * len(hit)) - 1], 3),
            "abs_yaw_aligned_p90": round(sorted(map(abs, pick("yaw_aligned")))[int(0.9 * len(hit)) - 1], 3),
            "same_plain_median": round(statistics.median(pick("same_plain")), 4),
            "same_aligned_median": round(statistics.median(pick("same_aligned")), 4),
            "same_aligned_below_0.45": sum(1 for v in pick("same_aligned") if v < 0.45),
            "other_aligned_at_or_above_0.45": sum(1 for v in pick("other_aligned") if v >= 0.45),
        }
    args.out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
