# Lab 3 — Registry and Docker

## 1. Registered model

The first registration of `food11` receives version 1; subsequent registrations increment the version number. A run artifact is an immutable training output. A registered model adds a shared name, version history, aliases and tags that support deployment independently of a run's other artifacts. Actual registrations are recorded in `lab-answers.md`.

## 2. Aliases

`champion` and `challenger` are chosen alias names, not built-in replacements with mandatory semantics. MLflow permits arbitrary aliases instead of fixed deprecated stages. A run can contain multiple models; registry versions identify individual deployable outputs, and an alias can move without altering their contents or callers' configuration.

## 3. Loading through the registry

`models:/food11@champion` resolves a named deployment decision and retrieves its artifact. Loading a `.pth` path couples serving to a machine and bypasses the registry. Assign `champion` to the desired version and restart inference to load that version. The API loads once at startup.

## 4. Build cache

Dependencies are installed after copying only the manifest and lockfile. A source-only edit invalidates the source copy and subsequent layers, while the expensive dependency-install layer stays cached. `--no-install-project` allows the builder to resolve third-party dependencies before source exists; the runtime imports the copied source through `PYTHONPATH`.

## 5. Image size

The multi-stage runtime copies the environment and source but omits uv and builder state. PyTorch and scientific libraries still dominate its size; multi-stage alone does not shrink those required libraries. Numeric sizes must be measured with `docker image inspect` and `docker history`; the execution record in `lab-answers.md` distinguishes measurements from expectations.

## 6. Docker ignore

Sending data, caches, Git history and model runs increases build-context transfer time and can leak credentials. Files only enlarge the final image if copied into a layer; context size and image size are different. Merely sending an excluded directory does not necessarily break a build. Copying a Windows `.venv` over the Linux environment can break executables and native libraries. `.dvc/config.local` also must never enter the image.

## 7. Host networking

Inside the container, loopback identifies that container. Docker Desktop's `host.docker.internal` resolves to the host gateway so the container can reach host services. Compose service DNS is used when MLflow runs in the same Compose network.

## 8. Restarting a container

A new container resolves and downloads the alias target during startup, provided MLflow and the artifact service are reachable. The image contains application code and dependencies; the trained weights and registry state are fetched at runtime. A registry failure prevents normal startup after bounded retries. `SKIP_MODEL=1` supports isolated CI smoke checks.

## 9. Releasing images

Push the built image to a registry and reference an immutable digest or a unique commit-SHA tag. A local `latest` tag cannot identify identical content across machines. Lab 5 builds, tests, and publishes `ghcr.io/husseinchr/mlops-lab-1:<commit-sha>`.
