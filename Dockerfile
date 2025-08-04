FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./

# install uv package and its dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && pip install --no-cache-dir uv==0.20.0 \
    && rm -rf /var/lib/apt/lists/*

RUN uv sync

COPY . .

RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]