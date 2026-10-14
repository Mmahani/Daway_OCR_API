# Daway Medicine OCR API

FastAPI service that reads a medicine package photo (Arabic / Latin) and returns the best-matching drug name. Consumed by the Daway Laravel backend.

## Run

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

## Endpoints

```bash
# health (model_loaded=false until the first OCR call downloads/loads the model)
curl localhost:8000/health

# OCR
curl -X POST localhost:8000/ocr/medicine \
  -F "file=@panadol.jpg;type=image/jpeg" \
  -H "X-API-Key: <OCR_API_KEY>"
```

Only `image/jpeg`, `image/jpg`, `image/png` are accepted. Error messages are Arabic by design.

## Env

| Var | Default | Purpose |
|---|---|---|
| `PORT` | `10000` | Docker serve port |
| `OCR_API_KEY` | _(empty = open)_ | When set, `X-API-Key` header is required (401 otherwise) |
| `MAX_IMAGE_MB` | `10` | Max upload size (413 beyond) |
| `OCR_TIMEOUT_SEC` | `120` | OCR processing timeout (504 beyond) |

Copy `.env.example` to `.env` for local use.

## Tests

```powershell
python -m unittest discover -s tests
```

Pure matching logic only — no model download needed.

## Docker

```bash
docker build -t daway-ocr .
docker run -p 10000:10000 -e OCR_API_KEY=secret daway-ocr
```

Runs as non-root `appuser`, single uvicorn worker (heavy model), `HEALTHCHECK` on `/health`.
