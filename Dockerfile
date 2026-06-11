# Streamlit gold dashboard — container image for Render (or any container host).
FROM python:3.12-slim

# libgomp1 is required by xgboost's wheel (OpenMP); slim doesn't ship it.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# uv (fast resolver/installer) from its official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev          # prod deps only (no pytest/ruff)
COPY . .

ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
EXPOSE 8501

# Render injects $PORT; fall back to 8501 locally. CORS/XSRF off for custom-domain websockets.
CMD uv run streamlit run streamlit_app.py \
    --server.port=${PORT:-8501} --server.address=0.0.0.0 \
    --server.enableCORS=false --server.enableXsrfProtection=false
