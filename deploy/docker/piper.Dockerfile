# Piper (GPL-3.0-or-later) runs here, in its own container; the repository only speaks HTTP to it.
# The voices are mounted at /voices (make models downloads them).
FROM python:3.13-slim
RUN pip install --no-cache-dir "piper-tts[http]==1.8.0" && useradd --system --uid 10001 piper
USER piper
EXPOSE 5100
CMD ["python", "-m", "piper.http_server", "-m", "en_US-ljspeech-medium", "--data-dir", "/voices", \
     "--host", "0.0.0.0", "--port", "5100"]
