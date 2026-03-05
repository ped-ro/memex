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

COPY sync_daemon.py .
# vault_import.py and sync.py are mounted from host at runtime

CMD ["python", "sync_daemon.py"]
