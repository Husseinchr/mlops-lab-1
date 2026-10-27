FROM ghcr.io/astral-sh/uv:0.12.13 AS uv-tool
FROM python:3.12-slim
ARG MLFLOW_VERSION=3.17.2
RUN --mount=type=secret,id=build_ca \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=from=uv-tool,source=/uv,target=/usr/local/bin/uv \
    if [ -f /run/secrets/build_ca ]; then \
      cat /etc/ssl/certs/ca-certificates.crt /run/secrets/build_ca > /tmp/build-ca.pem; \
      export SSL_CERT_FILE=/tmp/build-ca.pem; \
    fi; \
    uv pip install --system "mlflow==${MLFLOW_VERSION}" && rm -f /tmp/build-ca.pem
RUN mkdir -p /mlflow-data
EXPOSE 5000
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:////mlflow-data/mlflow.db", "--serve-artifacts", "--artifacts-destination", "/mlflow-data/mlruns", "--allowed-hosts", "mlflow:5000,localhost:5000,127.0.0.1:5000,host.docker.internal:5000"]
