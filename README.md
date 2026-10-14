# Daway Medicine OCR API

FastAPI service that reads a medicine package photo (Arabic / Latin) and returns the best-matching drug name. Consumed by the Daway Laravel backend.

## Env

| Var | Required | Default | Purpose |
|---|---|---|---|
| `API_KEY` | yes (prod) | _(empty = open)_ | Clients must send it as `X-API-Key` header |
| `ALLOWED_ORIGINS` | yes (browser clients) | _(empty)_ | Comma-separated CORS origins, e.g. `https://app.example.com,https://admin.example.com` |
| `DEBUG_OCR` | no | `false` | Set `true` to include `raw_texts` + `top_matches` in the response |
| `MAX_IMAGE_MB` | no | `5` | Max upload size (413 beyond) |
| `OCR_TIMEOUT_SEC` | no | `120` | OCR processing timeout (504 beyond) |
| `PORT` | no (Render injects it) | `10000` | Serve port |

Copy `.env.example` to `.env` for local use.

## Try it

```bash
# health (no key needed)
curl localhost:8000/health

# OCR
curl -X POST localhost:8000/ocr/medicine \
  -F "file=@panadol.jpg;type=image/jpeg" \
  -H "X-API-Key: <API_KEY>"
```

## Notes

- Only `image/jpeg`, `image/jpg`, `image/png` are accepted; error messages are Arabic.
- Images larger than 1600px (longest side) are downscaled before OCR to fit 512MB RAM.
- Drug list lives in `drugs.json` (same structure: `brand`, `brand_aliases`, `active_ingredients`).
- Tests: `python -m unittest discover -s tests` (no model download needed).
