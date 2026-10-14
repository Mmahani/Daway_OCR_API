FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 10000

# /health returns 200 even while the model is still loading
# ("starting"), so no restart loop during first-boot model download.
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD ["python", "-c", "import os,sys,urllib.request\nurl='http://127.0.0.1:'+os.environ.get('PORT','10000')+'/health'\ntry:\n    sys.exit(0 if urllib.request.urlopen(url,timeout=8).status==200 else 1)\nexcept Exception:\n    sys.exit(1)"]

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]