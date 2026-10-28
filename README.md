# Food-11 MLOps labs

Food image classification with versioned data, pretrained ResNet18 transfer learning,
MLflow experiment tracking and registry, FastAPI inference, Streamlit, Docker Compose,
and GitHub Actions.

Written answers and the execution record: [lab-answers.md](lab-answers.md).
Per-lab answers: [Lab 1](lab/lab1.md), [Lab 2](lab/lab2.md),
[Lab 3](lab/lab3.md), [Lab 4](lab/lab4.md), [Lab 5](lab/lab5.md).

## Install and retrieve data

Use Python 3.12, uv, DVC and Docker Desktop with Linux containers.

```powershell
uv sync --frozen
# Configure this machine's private DVC remote first; see lab/lab1.md.
dvc pull -r lab-local
uv run python src/food11/validate_data.py data/food11_processed_mini
```

The shared DagsHub remote remains configured as `origin`. This machine also uses
the course-approved local remote in `../dvc-storage`. Neither data nor credentials
are committed to Git. To regenerate from raw files:

```powershell
uv run python src/food11/data.py
uv run python src/food11/validate_data.py data/food11_processed_mini
dvc add data
dvc push -r lab-local
```

## Train and track experiments

Start only the tracking service first. An empty registry cannot yet serve a model.

```powershell
docker compose up -d mlflow
uv run python src/food11/train.py --dataset mini --epochs 5 --lr 0.001 --batch-size 32
```

Alternatively, outside Compose run a local server in another terminal:

```powershell
uv run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --serve-artifacts --artifacts-destination ./mlruns
```

Use only one tracking server on port 5000. Open http://127.0.0.1:5000 to compare runs.
Training validates the data before any optimization, freezes the pretrained backbone,
and logs epoch metrics, evaluation accuracy, class mapping, a PyTorch model and a
probability-serving adapter. Frozen features are cached privately by dataset hash.

Run all four comparison settings and both promotion candidates:

```powershell
uv run python scripts/run_experiments.py
```

For the course sequence, use `--phase compare` first, complete the serving and
Compose exercises, then use `--phase promote` for the two quality-gate candidates.
The comparison report is saved locally between phases.

Register an individual run and initialize its serving alias:

```powershell
uv run python src/food11/register.py --run-id <RUN_ID> --alias champion
```

The registry uses the run's `classifier` artifact, which wraps the required
`model` PyTorch artifact to return probabilities in numeric Food-11 class order.
`ImageFolder`'s alphabetical indices are explicitly remapped during training.

## Serve and orchestrate

```powershell
uv run uvicorn src.food11.serve:app --host 127.0.0.1 --port 8000
# Or, after registering champion in the Compose tracking server:
docker compose up --build -d
docker compose ps
```

On this workstation, HTTPS inspection requires a trusted build CA. The ignored
local override supplies it as an optional BuildKit secret:

```powershell
docker compose -f docker-compose.yml -f .verification/build.yml build
docker compose up -d --no-build
```

Normal CI builds need no local override. The CA is never copied into the runtime
image, and TLS verification stays enabled.

Open http://127.0.0.1:8501 to upload an image. The frontend calls inference inside
the Compose network; inference has no published host port. `GET /health` reports
liveness; `POST /predict` returns `category` and probability `confidence`.

For standalone container access to a host MLflow server on Docker Desktop:

```powershell
docker build -t food11-api:latest .
docker run --rm -p 8000:8000 -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 food11-api:latest
```

Apply the quality gate to a registered candidate, then restart inference if needed:

```powershell
uv run python src/food11/promote.py --version <VERSION>
docker compose restart inference
```

`docker compose down` preserves MLflow's named volume. `docker compose down -v`
deletes the database, registry and model artifacts; re-registration is then needed.

## Verify and release

```powershell
uv run ruff check src tests frontend scripts
uv run pytest -q
uv run python src/food11/validate_data.py data/food11_processed_mini
docker compose config --quiet
```

GitHub Actions runs lint/tests, then builds and smoke-tests the API with
`SKIP_MODEL=1`. Successful pushes to `main` publish the commit-tagged image:
`ghcr.io/husseinchr/mlops-lab-1:<commit-sha>`. CI does not train a model or fetch
private data. A repository administrator must require `test` and `smoke` on pull
requests in the branch protection settings. Code images and model aliases are
released independently.
