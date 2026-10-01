FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md run.py ./
COPY src ./src
COPY tests ./tests
COPY config ./config

RUN python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.6.0 torchaudio==2.6.0 \
    && python -m pip install --no-cache-dir .

CMD ["python", "run.py", "test"]