"""Evaluate local LFW pairs using SmartCheck's embedding pipeline; no DB writes.

Only aggregate scores are saved. Images and embeddings stay local/in memory.
This measures one-to-one matching, not liveness or production gallery accuracy.
"""
import argparse
import base64
from collections import Counter
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
THRESHOLD_NAMES = ('DUPLICATE_THRESHOLD', 'SELF_VERIFY_THRESHOLD',
                   'SAME_DEVICE_THRESHOLD', 'NEW_DEVICE_THRESHOLD',
                   'CONSISTENCY_THRESHOLD', 'CONTINUITY_THRESHOLD')


def read_pairs(path):
    """Read LFW pairs.txt (fold-count/pairs-per-class header or dev pair count)."""
    lines = [line.split() for line in path.read_text(encoding='utf-8').splitlines()
             if line.strip()]
    if not lines or len(lines[0]) not in (1, 2):
        raise ValueError('Expected an LFW pairs header')
    header = [int(n) for n in lines.pop(0)]
    if any(n <= 0 for n in header):
        raise ValueError('Pair counts must be positive')
    expected = 2 * math.prod(header)
    if len(lines) != expected:
        raise ValueError(f'Header expects {expected} pairs, found {len(lines)}')

    def image(name, index):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name) or int(index) < 1:
            raise ValueError('Invalid LFW image identifier')
        return f'{name}/{name}_{int(index):04d}.jpg'

    pairs = []
    for parts in lines:
        if len(parts) == 3:
            name, a, b = parts
            pairs.append((image(name, a), image(name, b), True))
        elif len(parts) == 4:
            name_a, a, name_b, b = parts
            if name_a == name_b:
                raise ValueError('Negative pair must contain different identities')
            pairs.append((image(name_a, a), image(name_b, b), False))
        else:
            raise ValueError('Expected 3 or 4 columns per pair')
    if sum(same for _, _, same in pairs) != expected // 2:
        raise ValueError('LFW pairs must have equal positive and negative counts')
    return pairs


def metrics(scores, threshold):
    positive = [score for same, score in scores if same]
    negative = [score for same, score in scores if not same]
    false_accepts = sum(s >= threshold for s in negative)
    false_rejects = sum(s < threshold for s in positive)
    return dict(threshold=threshold, genuine_pairs=len(positive), impostor_pairs=len(negative),
                false_accepts=false_accepts, false_rejects=false_rejects,
                false_accept_rate=false_accepts / len(negative) if negative else None,
                false_reject_rate=false_rejects / len(positive) if positive else None)


def evaluate(root, pairs, extract, similarity):
    cache, failures, scores = {}, Counter(), []
    excluded = Counter()
    for a, b, same in pairs:
        vectors = []
        for name in (a, b):
            if name not in cache:
                try:
                    vector = extract(base64.b64encode((root / name).read_bytes()).decode('ascii'))
                    if (len(vector) != 512 or not all(math.isfinite(x) for x in vector)
                            or not any(x != 0 for x in vector)):
                        raise ValueError('Invalid embedding')
                    cache[name] = vector
                except Exception as error:
                    failures[type(error).__name__] += 1
                    cache[name] = None
            vectors.append(cache[name])
        if any(vector is None for vector in vectors):
            excluded['genuine' if same else 'impostor'] += 1
            continue
        score = float(similarity(*vectors))
        if not math.isfinite(score):
            raise ValueError('Non-finite similarity score')
        scores.append((same, score))
    return scores, dict(excluded), dict(failures)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lfw-dir', type=Path, required=True, help='Extracted local LFW image directory')
    parser.add_argument('--pairs', type=Path, required=True, help='Local LFW pairs.txt or pairsDevTest.txt')
    parser.add_argument('--output', type=Path, required=True, help='New aggregate JSON file (will not overwrite)')
    parser.add_argument('--limit-per-class', type=int, help='Exploratory subset: first N pairs of each class')
    parser.add_argument('--thresholds', type=float, nargs='+', default=[0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9])
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Output already exists; choose a new filename')
    if args.limit_per_class is not None and args.limit_per_class < 1:
        parser.error('--limit-per-class must be positive')
    if any(not math.isfinite(t) or not -1 <= t <= 1 for t in args.thresholds):
        parser.error('Cosine thresholds must be finite and between -1 and 1')
    pairs = read_pairs(args.pairs)
    if args.limit_per_class:
        pairs = ([p for p in pairs if p[2]][:args.limit_per_class]
                 + [p for p in pairs if not p[2]][:args.limit_per_class])
    # Fail before model import, rather than silently downloading model weights.
    import os
    weights = Path(os.environ.get('DEEPFACE_HOME', str(Path.home()))) / '.deepface/weights/facenet512_weights.h5'
    if not weights.is_file():
        parser.error('Local facenet512_weights.h5 is missing; this script does not download weights')
    missing = {name for a, b, _ in pairs for name in (a, b) if not (args.lfw_dir / name).is_file()}
    if missing:
        parser.error(f'{len(missing)} input images are missing from --lfw-dir')

    sys.path.insert(0, str(ROOT))
    # Import the service only; never create a Flask app or Supabase client.
    from app.services import face_service as service
    current = {name: getattr(service, name) for name in THRESHOLD_NAMES}
    scores, excluded, failures = evaluate(args.lfw_dir, pairs, service.extract_embedding,
                                         service.cosine_similarity)
    report = dict(
        protocol='Local LFW pairwise evaluation at fixed thresholds; no threshold fitting',
        pipeline='SmartCheck CLAHE + DeepFace Facenet512 / opencv / cosine similarity >= threshold',
        runtime={'python': sys.version.split()[0], 'deepface': version('deepface'),
                 'tensorflow': version('tensorflow'), 'opencv': service.cv2.__version__,
                 'numpy': service.np.__version__},
        pairs_sha256=hashlib.sha256(args.pairs.read_bytes()).hexdigest(),
        exploratory_subset=bool(args.limit_per_class),
        requested_pairs=len(pairs), evaluated_pairs=len(scores),
        excluded_pairs_by_class=excluded, failed_images_by_error_type=failures,
        current_thresholds={name: metrics(scores, value) for name, value in current.items()},
        sweep=[metrics(scores, t) for t in sorted(set(args.thresholds))],
        limitations=[
            'Rates exclude failed image extractions; inspect exclusions before interpreting results.',
            'This is not a cross-validated LFW benchmark or a production threshold recommendation.',
            'One-to-one image pairs do not measure max-over-many embeddings or gallery duplicate search.',
            'Liveness, camera quality, EAR and Thai student population performance are not measured.',
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as out:
        json.dump(report, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write('\n')
    print(f'Evaluated {len(scores)}/{len(pairs)} pairs. Aggregate report: {args.output}')
    return 0 if any(same for same, _ in scores) and any(not same for same, _ in scores) else 2


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        print(f'Calibration failed: {error}', file=sys.stderr)
        raise SystemExit(2)
