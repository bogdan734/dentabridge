# The call interpreter (dentabridge-voice worker) for the cpu and cuda profiles. Build context: the
# repository root. CUDA=1 adds cuBLAS and cuDNN 9, which CTranslate2 needs on a GPU.
FROM python:3.13-slim
ARG CUDA=0
RUN pip install --no-cache-dir uv==0.11.19 && useradd --system --create-home --uid 10001 app
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_CACHE=1 UV_PYTHON_DOWNLOADS=never \
    PATH=/app/.venv/bin:$PATH \
    LD_LIBRARY_PATH=/app/.venv/lib/python3.13/site-packages/nvidia/cublas/lib:/app/.venv/lib/python3.13/site-packages/nvidia/cudnn/lib
COPY pyproject.toml uv.lock ./
COPY packages/core/pyproject.toml packages/core/
COPY packages/ml/pyproject.toml packages/ml/
COPY packages/pricing/pyproject.toml packages/pricing/
COPY packages/api/pyproject.toml packages/api/
COPY packages/voice/pyproject.toml packages/voice/
RUN uv sync --locked --no-dev --package dentabridge-voice --extra worker --no-install-workspace
RUN if [ "$CUDA" = 1 ]; then uv pip install "nvidia-cublas-cu12==12.*" "nvidia-cudnn-cu12==9.*"; fi
COPY packages packages
RUN uv sync --locked --no-dev --package dentabridge-voice --extra worker --inexact
USER app
ENTRYPOINT ["dentabridge-voice", "worker"]
