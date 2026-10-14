# AGENTS.md

Two-file FastAPI service: `app.py` (HTTP layer) + `ocr_service.py` (PaddleOCR + matching). No tests, linter, CI, or README.

## Run

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Docker (production parity): `CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}`, base `python:3.12-slim`.
- Image runs as non-root `appuser` with `--workers 1` (heavy model: no memory duplication). PaddleOCR models pre-download at build time (`RUN python -c ... PaddleOCR(...)` as appuser with `HOME=/home/appuser` so the cache is visible at runtime) + `ENV DISABLE_MODEL_SOURCE_CHECK=True`. `HEALTHCHECK` (exec-form python, no curl in slim) hits `/health` with `start-period=180s` for first-boot model download. Never build/run as root.
- Tests: `python -m unittest discover -s tests` (stdlib only, no pytest). `tests/test_matching.py` stubs `paddleocr` in `sys.modules` so no model download; never assert timeout timing via TestClient (it waits for the orphaned thread) — verify against live uvicorn instead.

## Gotchas

- Model init is lazy: `ocr_service.get_ocr()` singleton builds `PaddleOCR(lang="ar", ocr_version="PP-OCRv3", ...)` on first OCR call, not at import. Load failure raises `RuntimeError` (Arabic message) → 500. `GET /health` is a plain open `{"status":"ok"}` (no auth, no model state).
- `POST /ocr/medicine` runs OCR via `asyncio.to_thread` + `wait_for(OCR_TIMEOUT_SEC, default 120)` → 504 on timeout. The orphaned thread keeps running detached; safe because `process_medicine_image` cleans only its own enhanced-file copy. Do not use plain `await`/blocking call here — it stalls the event loop.
- `requirements.txt`: `numpy==2.2.6` (opencv-headless 5.x requires numpy>=2, so `numpy<2` breaks the resolver; paddle 3.2.0 verified working with numpy 2). Never pin `numpy==2.4.1` — present in local venv but not on PyPI. Keep `opencv-python-headless` (not `opencv-python`; Docker slim has no GUI libs, only `libgomp1 libglib2.0-0 libgl1`).
- `POST /ocr/medicine` accepts only `image/jpeg|image/jpg|image/png` as multipart `file`; error messages are Arabic by design. Empty file → 400.
- Upload guards in `app.py`: real decode check via `cv2.imdecode` (content-type alone is spoofable) → 400; size limit `MAX_IMAGE_MB` (default 5) → 413; `API_KEY` env (renamed from `OCR_API_KEY`) — `X-API-Key` header required on `/ocr/medicine` (401), `/health` open. CORS origins from `ALLOWED_ORIGINS` (comma-separated). Unexpected errors → generic Arabic 500, details only in logs via `logger.exception`. See `.env.example`.
- Both `app.py:133` and `ocr_service.py:344` use `tempfile.NamedTemporaryFile(delete=False)` + manual `os.remove` in `finally`. Preserve this pattern (Windows file locking); never switch to `delete=True` with OpenCV/Paddle reading the path.
- Preprocess (`ocr_service.py`): upscale capped at `scale = min(3, 1600/max(h,w))`, and any image with longest side >1600px is downscaled before OCR (`downscale_if_needed`) for 512MB RAM. Do not raise the 1600 cap without checking Render memory.
- `drug_database` loads from `drugs.json` (same dir as `ocr_service.py`) at startup — same structure (`brand`, `brand_aliases`, `active_ingredients`). To add a drug, edit the JSON.
- `DEBUG_OCR=true` adds `raw_texts` + `top_matches` (top 3) to the response; otherwise those keys are absent. Never change `get_top_matches` logic or the 0.80/0.65 thresholds.
- Match thresholds in `build_ocr_response`: `>= 0.80` confirmed, `>= 0.65` needs_confirmation, else uncertain/not_found. Response envelope (`source`, `intent: search_medicine`, `drug_name`, `symptoms: []`, `ocr_success`, `status`, `best_candidate`, `match_score`, `message`) is consumed by the Laravel backend — keep keys stable. Matching normalizes أ/إ/آ → ا with word-level guard (`len>=4`, length diff `<=1`).
