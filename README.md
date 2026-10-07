# AI Image Forensics

A modular FastAPI application for collecting explainable image-origin evidence. It validates and decodes JPEG, PNG, WebP and TIFF uploads, extracts available metadata, calculates visual statistics, runs Sightengine's hosted AI-image detector when configured (otherwise a local classifier), and stores JSON analysis reports without keeping uploaded image bytes.

With Sightengine credentials configured, the app returns a clear `LIKELY AI-GENERATED` or `LIKELY GENUINE` result and the provider's uncalibrated model score. Otherwise it uses the local Nonescape Mini model if its weights are installed. Without either detector it returns `INCONCLUSIVE`. No result proves image origin; detector accuracy has not been independently evaluated by this project.

## Architecture

```text
Browser dashboard → FastAPI upload API → decode and validate
                                       ├─ metadata / filename checks
                                       ├─ image feature extraction
                                       ├─ Sightengine genai API (optional)
                                       ├─ local Nonescape Mini image classifier (fallback)
                                       └─ evidence report → SQLite + JSON response
```

The stable analysis API is in `app/api`. Service modules separate upload security, metadata, filename analysis, image features, classification policy, explanations, and reports. `app/ml` loads the local Nonescape Mini model once; it never downloads weights automatically. If Sightengine is configured, image bytes are sent to its API for analysis; a failed request falls back to the local model. The optional CPU baseline uses NumPy/SciPy measurements and a calibrated logistic regression model.

## Features

- Content-based format verification, extension mismatch warning, decoded-image check, pixel and byte limits, safe filename display, SHA-256, and transient upload handling.
- Best-effort EXIF, XMP, IPTC and ICC extraction, with optional ExifTool enrichment and fallback. GPS coordinates are never returned; only presence is recorded.
- Filename clues with weak-evidence language, timestamp ordering checks and software markers.
- Measured noise residual, frequency spectrum, gradient/edge, texture, color and JPEG quantization features.
- Local pretrained AI-generated-vs-authentic model with a clear top-class result, visible raw model score, and explicit calibration status.
- Optional Sightengine hosted AI-generated image detection with local fallback and an explicit upload disclosure.
- Responsive browser UI, image preview, report JSON download, evidence list, and metadata/measurement tables.
- SQLite report persistence (no uploaded images), Docker, Render Blueprint, and GitHub Actions configuration.

## Local development

Use Python 3.11 or newer.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://127.0.0.1:8000`. The API documentation is at `/api/docs`; health and version are `/api/health` and `/api/version`.

Download the model before starting the app if you want image classification:

```powershell
$modelUrl = "https://huggingface.co/e3ntity/nonescape-v0/resolve/dd9d70c10cd0f6823e2af87d553217c25fe00b3d/nonescape-mini-v0.safetensors"
Invoke-WebRequest -Uri $modelUrl -OutFile "models/nonescape-mini-v0.safetensors"
Get-FileHash -Algorithm SHA256 "models/nonescape-mini-v0.safetensors"
```

Compare the downloaded file's SHA-256 to the expected value in [models/README.md](models/README.md). For hosted detection, set `SIGHTENGINE_API_USER` and `SIGHTENGINE_API_SECRET` in `.env`; the configured app sends image bytes to Sightengine. When credentials are absent, the local detector runs without sending the image externally.

## API usage

```powershell
curl.exe -F "file=@C:\path\to\image.jpg" http://127.0.0.1:8000/api/analyze
```

The response includes `analysis_id`, SHA-256, file properties, metadata, filename assessment, categorized image measurements, evidence, model state, warnings, and limitations. Sightengine's `type.ai_generated` score is shown as an uncalibrated model score and mapped to `LIKELY_AI_GENERATED` or `LIKELY_AUTHENTIC`; it is not a validated probability. Retrieve a stored result with `GET /api/report/{analysis_id}`. Invalid content returns 400, unsupported extensions/formats return 415, oversized uploads return 413, and missing multipart fields return 422. The browser dashboard is served from `/`.

## Dataset and training

To split unsplit `real/` and `ai/` directories before training, run:

```powershell
python -m scripts.split_dataset --input C:\data\unsplit --output C:\data\image-dataset --train 0.70 --validation 0.15 --test 0.15 --seed 17
```

The splitter clusters exact and pHash-near images before assigning groups, rejects cross-label duplicates, and requires enough independent groups to populate all splits. Group by photographer, generator or source before running it where those identifiers are available; perceptual hashes alone cannot identify every related image.

The tools require images supplied by you. Read [datasets/README.md](datasets/README.md) for structure, provenance, licensing and split guidance. Keep source groups and near duplicates within a single split. The training command verifies exact SHA-256 and close perceptual hashes across the three splits before fitting.

```powershell
python -m scripts.train --dataset C:\data\image-dataset --output models/baseline.joblib --model-version project-1
```

The trainer fits only `train/real` and `train/ai`, optionally calibrates probabilities on `validation`, selects a balanced-accuracy decision threshold on `validation`, and leaves `test` untouched. Evaluation and benchmark commands use that saved threshold by default; `--threshold` overrides it. The app keeps an uncalibrated raw model score separate from calibrated AI probability. Uncalibrated models report their top class as likely; calibrated models use conservative inconclusive bands. The detector is binary and does not train a manipulated-image class. Joblib artifacts are executable pickle files: only load artifacts produced by a trusted training environment.

## Evaluation

```powershell
python -m scripts.evaluate --dataset C:\data\image-dataset --model models/baseline.joblib --output reports/evaluation --threshold 0.5
python -m scripts.benchmark --dataset C:\data\benchmark --model models/baseline.joblib --output reports/benchmark.json --robustness
python -m scripts.extract_features --dataset C:\data\image-dataset --output reports/features.csv
```

`evaluate.py` requires the isolated `test` split and writes `evaluation_results.json` and `confusion_matrix.png`. It reports accuracy, per-class precision/recall/F1, real-class specificity, ROC-AUC, a bootstrap accuracy interval, confusion matrix, model/dataset versions, and threshold comparisons. ROC-AUC is unavailable if test data contains only one class. The 90% indicator is derived from measured accuracy; it is never a preset claim. Manipulated examples are not included in this binary metric. `benchmark.py` reports category metrics for folders under `test/<category>/{real,ai}`. Its `--robustness` option generates JPEG recompression, resize, crop, color, blur, metadata-stripping, and social-style compression variants from held-out `test/real` and `test/ai` samples. Add real screenshots under `test/screenshots/{real,ai}`; screenshots are not simulated. Keep category variants source-grouped in test and pair AI-only unseen-generator checks with real controls before comparing performance.

No evaluation metrics are published by this repository: no dataset or trained model is included. Performance and 90% accuracy remain unknown until you run evaluation on licensed, representative held-out data. A strong score on one dataset does not establish performance on unseen generators or screenshots, edits, crops, resizing, recompression, color changes, blur or stripped metadata. Prepare those categories in the held-out benchmark and report them separately.

## Tests and checks

```powershell
python -m pytest -q
ruff check app scripts tests
ruff format app scripts tests
ruff format --check app scripts tests
```

CI runs Ruff linting and formatting, pytest, and an API import check on pushes and pull requests without downloading model weights. Tests use synthetic images and do not claim detector accuracy.

## Docker

```powershell
docker build -t ai-image-forensics .
docker run --rm -p 8000:8000 -e PORT=8000 ai-image-forensics
```

The image runs as a non-root user and serves on `0.0.0.0`. For persistent results and model artifacts, mount or provision storage and set `DATABASE_URL` and `MODEL_PATH` accordingly.

## Render deployment

Push the repository to GitHub and create a Render Blueprint from `render.yaml`, or create a Docker web service from this repository. The Dockerfile starts Uvicorn using the `PORT` environment variable and health checks use `/api/health`. Render's Docker runtime uses the Dockerfile command by default; see [Render Docker deployment](https://render.com/docs/docker) and the [Blueprint reference](https://render.com/docs/blueprint-spec).

The Blueprint marks `SIGHTENGINE_API_USER` and `SIGHTENGINE_API_SECRET` as dashboard-provided values. Enter them during initial Blueprint creation. For an existing Render service, add them under its Environment settings; Blueprint syncs do not prompt again for `sync: false` values.

SQLite and the container filesystem are not durable across all service lifecycle events. For persistent report history, attach a supported persistent disk or replace `app/services/repository.py` with a PostgreSQL-backed adapter. Keep trained model artifacts available at `MODEL_PATH`; the repository intentionally does not include them.

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `PORT` | `8000` | HTTP listen port |
| `MAX_UPLOAD_MB` | `15` | Maximum request file bytes |
| `MAX_IMAGE_PIXELS` | `20000000` | Maximum decoded image pixels |
| `NONESCAPE_MODEL_PATH` | `models/nonescape-mini-v0.safetensors` | Local pretrained AI-generated image detector |
| `SIGHTENGINE_API_USER` | unset | Sightengine API user ID; enables hosted detection when used with the secret |
| `SIGHTENGINE_API_SECRET` | unset | Sightengine API secret; keep it in the ignored `.env` file |
| `MODEL_PATH` | `models/baseline.joblib` | Optional trusted trained model artifact |
| `DATABASE_URL` | `sqlite:///reports/analyses.sqlite3` | SQLite result database URL |
| `LOG_LEVEL` | `INFO` | Structured log verbosity |
| `APP_VERSION` | `0.1.0` | API version identifier |

## GitHub

```powershell
git init
git add .
git commit -m "Build image forensics baseline"
git branch -M main
git remote add origin https://github.com/OWNER/REPOSITORY.git
git push -u origin main
```

`.gitignore` excludes local models, datasets, reports, databases, uploads and environment files. Review licenses and data sensitivity before adding any sample image or model artifact.

## Security and limitations

- An ASGI request-body limit is checked before multipart parsing, then bytes are bounded again before decoding. Framework upload spooling is temporary and cleaned after request handling; images are not persisted. The UI preview uses a browser object URL that is revoked on replacement.
- The decoder is limited to allowed raster image formats and pixel counts. It is not a malware scanner; keep dependencies patched and run behind standard network/rate limits for public deployment.
- Report JSON retains extracted metadata such as camera and software names, but not exact GPS coordinates or uploaded pixels. Restrict report database access and retention to the intended audience.
- Metadata can be forged or stripped. Filename, compression, noise and frequency features are not individually reliable origin evidence.
- The pretrained model score is uncalibrated for this application and may fail on sources outside its training data. A clear top-class result is an estimate, not proof that an image is fake or genuine.
- When Sightengine credentials are configured, uploaded image bytes are sent to Sightengine; review its current [API documentation](https://www.sightengine.com/docs/ai-generated-image-detection) and data terms before deployment. Provider scores are uncalibrated in this application.
- Manipulation detection, source authentication, adversarial benchmark transformation generation and multi-class evaluation are future work. IPTC extraction is best-effort through Pillow/ExifTool, depending on installed support. No result proves that an image is authentic or AI-generated.

## Project layout

```text
app/               FastAPI API, configuration, security, analysis and ML modules
frontend/          Static responsive browser dashboard
models/            Local model artifact instructions (weights are downloaded separately)
datasets/          Dataset schema and provenance guidance
scripts/           Dataset splitting, training, evaluation, benchmark and feature export CLIs
tests/             API, security, metadata, feature, model and dataset tests
.github/workflows/ CI checks
```
