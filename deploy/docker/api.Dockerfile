# The API (dentabridge-api serve). Build context: the repository root.
FROM python:3.13-slim
RUN pip install --no-cache-dir uv==0.11.19 && useradd --system --uid 10001 app
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_CACHE=1 UV_PYTHON_DOWNLOADS=never \
    PATH=/app/.venv/bin:$PATH
# the locked dependencies first, so a code change does not reinstall them
COPY pyproject.toml uv.lock ./
COPY packages/core/pyproject.toml packages/core/
COPY packages/ml/pyproject.toml packages/ml/
COPY packages/pricing/pyproject.toml packages/pricing/
COPY packages/api/pyproject.toml packages/api/
COPY packages/voice/pyproject.toml packages/voice/
RUN uv sync --locked --no-dev --package dentabridge-api --no-install-workspace
# the code reads its data next to the sources (seed/, packages/*/data, the replays), so the
# workspace stays editable in /app
COPY packages packages
COPY seed seed
COPY deploy/release.json deploy/release.json
RUN uv sync --locked --no-dev --package dentabridge-api && mkdir var && chown app var
USER app
EXPOSE 8000
CMD ["dentabridge-api", "serve", "--host", "0.0.0.0", "--port", "8000"]
