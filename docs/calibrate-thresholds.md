# Local face threshold evaluation

`scripts/calibrate_thresholds.py` measures false matches and missed matches at
the application's current thresholds, including DUPLICATE (0.65) and SELF_VERIFY
(0.80). It also reports a fixed threshold sweep. It does not change thresholds.

## Inputs and command

Use the project virtual environment and place an extracted LFW dataset and its
official pairs file on your machine. Expected layout:

```text
lfw/
  Person_Name/
    Person_Name_0001.jpg
    Person_Name_0002.jpg
pairs.txt
```

The script accepts standard LFW `pairs.txt` and development pairs files. It checks
the declared number of pairs, class balance and missing image files. FaceNet512
weights must already exist in `.deepface/weights/facenet512_weights.h5` under the
user home (or `DEEPFACE_HOME`). It does not download a dataset or weights.

PowerShell, from the repository root:

```powershell
.\venv\Scripts\python.exe scripts/calibrate_thresholds.py --lfw-dir C:\datasets\lfw --pairs C:\datasets\pairs.txt --output .build\lfw-thresholds.json
```

For a quick exploratory run, add `--limit-per-class 10`. This uses the first 10
genuine and first 10 impostor pairs, not a representative random sample. Use a
new output filename for each run; existing files are never overwritten.

## Report

- **false_accept_rate**: fraction of evaluated different-person pairs incorrectly
  accepted as a match (similarity >= threshold).
- **false_reject_rate**: fraction of evaluated same-person pairs rejected.
- Counts and exclusions accompany rates. Failed detections are excluded from
  both denominators and reported separately, not silently treated as mismatches.
- If either class has no evaluable pairs, its rate is `null` and the command exits
  with status 2 after saving the report. Otherwise it exits with status 0.
- The report records runtime versions, input-pairs SHA-256, thresholds and counts.
  It contains no images, identity names, or embeddings. Embeddings are cached only
  in RAM for the duration of the process. The script imports the face service but
  never creates the Flask application or initializes a Supabase client.

This is a fixed-threshold, pairwise measurement, not the official cross-validated
LFW benchmark. Do not tune on this report and present the same pairs as independent
validation. Production uses maximum similarity over several embeddings and, for
duplicate detection, multiple people; those error rates differ from pairwise
rates. LFW does not validate liveness, EAR, phone-camera behavior or performance
on the actual student population.

## Validation performed

Unit tests cover pair parsing, malformed inputs, threshold boundaries, missing
classes, embedding caching, invalid embeddings and extraction failures. A local
smoke run processed two pairs from existing `test_images` through the actual
FaceNet512 model. That smoke run checks execution only: its tiny, manually chosen
sample is **not LFW and is not evidence for changing any production threshold**.
The LFW measurement remains pending until the dataset is provided locally.
