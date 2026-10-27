# Lab 2 — Training and MLflow

## 1. Dependency files

`pyproject.toml` now declares MLflow, PyTorch, torchvision and scikit-learn, in addition to the serving dependencies. `uv.lock` pins their resolved versions and transitive dependencies for reproducible installations. An explicit CPU PyTorch index keeps Linux CI and Docker from downloading CUDA libraries. The project exposes the `food11` package from `src/`.

## 2. Metadata and artifacts

`--backend-store-uri sqlite:///mlflow.db` selects the SQL database for experiments, runs, parameters, metric histories, tags, and registry records. Artifacts are larger files such as serialized models and class mappings. This implementation uses `--serve-artifacts --artifacts-destination ./mlruns`: clients upload and download through MLflow's HTTP API. This is necessary when a Linux container cannot access a Windows artifact path. `--default-artifact-root` instead sets the artifact URI recorded for newly created experiments; a direct filesystem URI requires every client to access that filesystem.

Reference: [MLflow tracking server and proxied artifacts](https://www.mlflow.org/docs/latest/self-hosting/architecture/tracking-server/).

## 3. Ignore tracking outputs

The live database and artifact directory belong to the tracking service, which manages their consistency and lifecycle. Git would collect large, changing binaries; DVC would snapshot implementation details of an active service rather than the experiment semantics. Both are explicitly ignored. Model selection is recorded in MLflow's registry.

## 4. Creating an experiment

`set_experiment("food11")` creates the experiment if it does not already exist and selects it for subsequent runs. Repeating it selects the same experiment rather than duplicating it.

## 5. Parameters versus metrics

A parameter describes a fixed run input, such as learning rate. A metric is a measured numeric result, such as validation loss. The metric step identifies the epoch and preserves the history for charts; changing a parameter's value inside the same run would make the run configuration ambiguous.

## 6. Logged model location

Each run records a PyTorch `model` artifact and a `classifier` pyfunc adapter. The adapter applies softmax so the serving API receives probabilities, while preserving the numeric Food-11 class order. The experiment artifact URI uses `mlflow-artifacts:/...`; its files physically live under the tracking server's `mlruns/` directory, or `/mlflow-data/mlruns` in Compose. The SQLite database contains references and metadata, not the model weights themselves.

## 7–9. Comparison and best run

Measured runs, the selected run ID, and the relationship between learning rate, batch size, and accuracy are recorded in the experiment results section of `lab-answers.md` after execution. Higher learning rate is not inherently better: it can converge faster but overshoot; too small a rate may underfit within five epochs. Accuracy is maximized, while loss is minimized. Model selection uses validation accuracy, with evaluation accuracy reported separately.

The frozen pretrained ResNet18 backbone is evaluated once per dataset content hash; cached features are reused across runs. Only the final eleven-class layer is trained. This is transfer learning, makes the CPU experiment practical, and preserves the full network for inference. Dataset content hashes and test sizes are logged to prevent comparisons across different data snapshots.
