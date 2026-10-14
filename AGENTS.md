# AGENTS.md

Two-file FastAPI service: `app.py` (HTTP layer) + `ocr_service.py` (PaddleOCR + matching). No tests, linter, CI, or README.

## Run

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Docker (production parity): `CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}`, base `python:3.12-slim`.
- Image runs as non-root `appuser` with `--workers 1` (heavy model: no memory duplication). `HEALTHCHECK` (exec-form python, no curl in slim) hits `/health` with `start-period=180s` for first-boot model download. Never build/run as root.
- Tests: `python -m unittest discover -s tests` (stdlib only, no pytest). `tests/test_matching.py` stubs `paddleocr` in `sys.modules` so no model download; never assert timeout timing via TestClient (it waits for the orphaned thread) — verify against live uvicorn instead.

## Gotchas

- Model init is lazy: `ocr_service.get_ocr()` singleton builds `PaddleOCR(lang="ar", ocr_version="PP-OCRv3", ...)` on first OCR call, not at import. Load failure raises `RuntimeError` (Arabic message) → 500. `is_model_loaded()` backs `GET /health` (`starting`/`ok`); `GET /` stays a dumb liveness ping.
- `POST /ocr/medicine` runs OCR via `asyncio.to_thread` + `wait_for(OCR_TIMEOUT_SEC, default 120)` → 504 on timeout. The orphaned thread keeps running detached; safe because `process_medicine_image` cleans only its own enhanced-file copy. Do not use plain `await`/blocking call here — it stalls the event loop.
- `requirements.txt`: `numpy==2.2.6` (opencv-headless 5.x requires numpy>=2, so `numpy<2` breaks the resolver; paddle 3.2.0 verified working with numpy 2). Never pin `numpy==2.4.1` — present in local venv but not on PyPI. Keep `opencv-python-headless` (not `opencv-python`; Docker slim has no GUI libs, only `libgomp1 libglib2.0-0 libgl1`).
- `POST /ocr/medicine` accepts only `image/jpeg|image/jpg|image/png` as multipart `file`; error messages are Arabic by design. Empty file → 400.
- Upload guards in `app.py`: real decode check via `cv2.imdecode` (content-type alone is spoofable) → 400; size limit `MAX_IMAGE_MB` (default 10) → 413; optional `OCR_API_KEY` env — when set, clients must send `X-API-Key` header or get 401; when unset, endpoint stays open (dev mode). See `.env.example`.
- Both `app.py:133` and `ocr_service.py:344` use `tempfile.NamedTemporaryFile(delete=False)` + manual `os.remove` in `finally`. Preserve this pattern (Windows file locking); never switch to `delete=True` with OpenCV/Paddle reading the path.
- Preprocess (`ocr_service.py:316`: 3x upscale + CLAHE) runs only if first OCR pass is not `confirmed`. Final result is whichever pass has the higher `match_score`.
- Match thresholds in `build_ocr_response`: `>= 0.80` confirmed, `>= 0.65` needs_confirmation, else uncertain/not_found. Response envelope (`source`, `intent: search_medicine`, `drug_name`, `symptoms: []`, `ocr_success`, `status`, `best_candidate`, `match_score`, `message`) is consumed by the Laravel backend — keep keys stable.
- `drug_database` (`ocr_service.py:50`) is a hardcoded temp stand-in for future Laravel data. To add a drug, append `{brand, brand_aliases, active_ingredients}` with both Latin and Arabic aliases; matching normalizes أ/إ/آ → ا and uses `SequenceMatcher` with word-level guard (`len>=4`, length diff `<=1`).
