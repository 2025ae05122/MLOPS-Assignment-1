# syntax=docker/dockerfile:1

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/opt/app/artifacts/heart_disease_pipeline.joblib
ENV LOG_LEVEL=INFO

WORKDIR /opt/app

RUN addgroup --system appgroup \
    && adduser --system \
       --ingroup appgroup \
       appuser

COPY requirements.txt .

# The corporate CA bundle is optional.
# Local corporate builds pass it as a temporary BuildKit secret.
# GitHub Actions builds use the image's standard public CA trust.
RUN --mount=type=secret,id=corporate_ca \
    if [ -s /run/secrets/corporate_ca ]; then \
        export PIP_CERT=/run/secrets/corporate_ca; \
        echo "Using supplied corporate CA bundle for pip"; \
    else \
        echo "Using standard public CA trust for pip"; \
    fi; \
    python -m pip install \
        --no-cache-dir \
        --upgrade pip; \
    python -m pip install \
        --no-cache-dir \
        -r requirements.txt

COPY api ./api
COPY artifacts ./artifacts

RUN chown -R appuser:appgroup /opt/app

USER appuser

EXPOSE 8000

HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=15s \
    --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]