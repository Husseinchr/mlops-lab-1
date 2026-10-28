FROM ghcr.io/astral-sh/uv:0.12.13 AS uv-tool
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN --mount=type=secret,id=build_ca \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=from=uv-tool,source=/uv,target=/usr/local/bin/uv \
    if [ -f /run/secrets/build_ca ]; then \
      cat /etc/ssl/certs/ca-certificates.crt /run/secrets/build_ca > /tmp/build-ca.pem; \
      export SSL_CERT_FILE=/tmp/build-ca.pem; \
    fi; \
    UV_LOCK_TIMEOUT=3600 uv pip install --system -r requirements.txt && rm -f /tmp/build-ca.pem
COPY app.py .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
