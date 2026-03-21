FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    psycopg2-binary \
    python-frontmatter \
    tiktoken \
    requests \
    schedule

COPY sync/sync_daemon.py .
COPY sync.py ./host/sync.py
COPY vault_import.py ./host/vault_import.py

CMD ["python", "sync_daemon.py"]
