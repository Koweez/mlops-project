FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY server.py /app/server.py
COPY utils.py /app/utils.py
COPY models/ /app/models

EXPOSE 8000

HEALTHCHECK CMD curl --fail http://localhost:8000/_stcore/health

ENTRYPOINT ["uvicorn", "server:app"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
