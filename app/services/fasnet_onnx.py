"""DeepFace's Fasnet anti-spoof on ONNX Runtime, without torch or DeepFace.

Same two MiniVision Silent-Face models, exported from the same weights by
scripts/export_fasnet_onnx.py, with the preprocessing of
deepface/models/spoofing/FasNet.py (deepface 0.0.93), read from the installed
source:

  * input: the full BGR frame and the face box (x, y, w, h);
  * model 1 MiniFASNetV2 on a crop of the box scaled 2.7x, model 2
    MiniFASNetV1SE on a crop scaled 4.0x (`_get_new_box`: scale capped so the
    crop fits, then shifted back inside the frame; slice includes the end pixel);
  * each crop resized to 80x80 with cv2.resize (bilinear), kept BGR, 0-255
    floats (to_tensor deliberately drops the /255), CHW, batch of 1;
  * softmax of each model's 3 logits, summed; label = argmax; real if label 1;
    score = summed probability of the label / 2.
"""
import os
import threading

import cv2
import numpy as np

_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_FILES = (("fasnet_minifasnetv2_2.7_80x80.onnx", 2.7), ("fasnet_minifasnetv1se_4.0_80x80.onnx", 4.0))
_sessions = None
_lock = threading.Lock()


def available():
    return all(os.path.exists(os.path.join(_DIR, name)) for name, _ in MODEL_FILES)


def _get_sessions():
    global _sessions
    if _sessions is None:
        with _lock:
            if _sessions is None:
                import onnxruntime as ort
                _sessions = [(ort.InferenceSession(os.path.join(_DIR, name), providers=["CPUExecutionProvider"]), scale)
                             for name, scale in MODEL_FILES]
    return _sessions


def _get_new_box(src_w, src_h, bbox, scale):
    """deepface FasNet._get_new_box, unchanged."""
    x, y, box_w, box_h = bbox
    scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))
    new_width = box_w * scale
    new_height = box_h * scale
    center_x, center_y = box_w / 2 + x, box_h / 2 + y
    left_top_x = center_x - new_width / 2
    left_top_y = center_y - new_height / 2
    right_bottom_x = center_x + new_width / 2
    right_bottom_y = center_y + new_height / 2
    if left_top_x < 0:
        right_bottom_x -= left_top_x
        left_top_x = 0
    if left_top_y < 0:
        right_bottom_y -= left_top_y
        left_top_y = 0
    if right_bottom_x > src_w - 1:
        left_top_x -= right_bottom_x - src_w + 1
        right_bottom_x = src_w - 1
    if right_bottom_y > src_h - 1:
        left_top_y -= right_bottom_y - src_h + 1
        right_bottom_y = src_h - 1
    return int(left_top_x), int(left_top_y), int(right_bottom_x), int(right_bottom_y)


def crop(img, bbox, scale, out_w=80, out_h=80):
    """deepface FasNet.crop, unchanged."""
    src_h, src_w, _ = np.shape(img)
    x1, y1, x2, y2 = _get_new_box(src_w, src_h, bbox, scale)
    return cv2.resize(img[y1:y2 + 1, x1:x2 + 1], (out_w, out_h))


def _softmax(logits):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def probabilities(img_bgr, facial_area):
    """Per-model softmax outputs, [(3,), (3,)] — for parity tests."""
    out = []
    for session, scale in _get_sessions():
        blob = crop(img_bgr, facial_area, scale).transpose(2, 0, 1)[np.newaxis].astype(np.float32)
        logits = session.run(None, {session.get_inputs()[0].name: blob})[0]
        out.append(_softmax(logits.astype(np.float64))[0])
    return out


def analyze(img_bgr, facial_area):
    """(is_real, score) exactly as deepface Fasnet.analyze returns them."""
    prediction = np.sum(probabilities(img_bgr, facial_area), axis=0)
    label = int(np.argmax(prediction))
    return label == 1, float(prediction[label] / 2)
