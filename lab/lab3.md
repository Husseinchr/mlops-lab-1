# Lab 3 — Registry and Docker

## 1. Registered model

The selected run `0a9fcf208f4949548e03aacafc43aa65` was registered as **`food11` version 1** and assigned `champion`. Subsequent registrations increment the version number. A run artifact is an immutable training output. A registered model adds a shared name, version history, aliases and tags that support deployment independently of a run's other artifacts.

## 2. Aliases

`champion` and `challenger` are chosen alias names, not built-in replacements with mandatory semantics. MLflow permits arbitrary aliases instead of fixed deprecated stages. A run can contain multiple models; registry versions identify individual deployable outputs, and an alias can move without altering their contents or callers' configuration.

## 3. Loading through the registry

`models:/food11@champion` resolves a named deployment decision and retrieves its artifact. Loading a `.pth` path couples serving to a machine and bypasses the registry. Assign `champion` to the desired version and restart inference to load that version. The API loads once at startup.

The real local API returned HTTP 200 for health and classified the Bread fixture as `Bread` with confidence `0.9152076840400696`. Invalid image content returned HTTP 400. This exercised the registered probability adapter and its actual trained weights, rather than the unit-test fake.

## 4. Build cache

Dependencies are installed after copying only the manifest and lockfile. A source-only edit invalidates the source copy and subsequent layers, while the expensive dependency-install layer stays cached. `--no-install-project` allows the builder to resolve third-party dependencies before source exists; the runtime imports the copied source through `PYTHONPATH`.

## 5. Image size

Measured with `docker image inspect`, the multi-stage image is **1,783,322,569 bytes** (1.783 GB), versus **1,839,700,717 bytes** (1.840 GB) for a matching single-stage build. The reduction is 56,378,148 bytes, approximately **56.38 MB or 3.06%**. Both use the same dependency environment and source; the single-stage comparison also keeps uv. `docker history` shows that copying `.venv` is the largest runtime layer at about **1.66 GB**. The required PyTorch and scientific libraries dominate, so multi-stage does not magically remove their size. Separating training-only dependencies from serving dependencies could reduce it further.

## 6. Docker ignore

Sending data, caches, Git history and model runs increases build-context transfer time and can leak credentials. Files only enlarge the final image if copied into a layer; context size and image size are different. Merely sending an excluded directory does not necessarily break a build. Copying a Windows `.venv` over the Linux environment can break executables and native libraries. `.dvc/config.local` also must never enter the image.

## 7. Host networking

Inside the container, loopback identifies that container. Docker Desktop's `host.docker.internal` resolves to the host gateway so the container can reach host services. Compose service DNS is used when MLflow runs in the same Compose network.

## 8. Restarting a container

A new container resolves and downloads the alias target during startup, provided MLflow and the artifact service are reachable. The image contains application code and dependencies; the trained weights and registry state are fetched at runtime. A registry failure prevents normal startup after bounded retries. `SKIP_MODEL=1` supports isolated CI smoke checks.

The Linux image passed a model-free health check (HTTP 200), real model inference (Bread, confidence `0.9152076244354248`) and invalid-upload rejection (HTTP 400). The small floating-point difference from Windows does not change the predicted category.

Stopping/removing that API container and creating a new one from the same image also passed real prediction without a rebuild. Both containers used image ID `92e087c7188f33231219b9dd405df8017cf47bfa11df102e0c1d9247e8aab6f0` and fetched their weights from the host tracking server.

## 9. Releasing images

Push the built image to a registry and reference an immutable digest or a unique commit-SHA tag. A local `latest` tag cannot identify identical content across machines. Lab 5 builds, tests, and publishes `ghcr.io/husseinchr/mlops-lab-1:<commit-sha>`.
