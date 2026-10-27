# Lab 4 — Docker Compose

The subject's references to `Staging` are interpreted as the `champion` alias, consistently with Labs 3 and 5.

## 1. Container filesystem

Stopping a container preserves its writable layer; removing it deletes that layer. A newly created container from the same image starts with a fresh tracking database unless persistent storage is mounted. Registering nothing and observing an empty UI before and after does not alone prove persistence: a marker or model must be created to distinguish the cases.

## 2. Named volumes

A named volume survives container replacement, avoids host-specific paths, and works naturally across Docker Desktop and Linux. A bind mount also works if the directory permissions and paths are correct, but couples the setup to the host layout. Neither is an off-machine backup.

## 3. Service DNS

Compose attaches the services to a common private network. Its DNS maps `mlflow` to that service's address. An unrelated standalone container has no membership in that network or its service-name namespace.

## 4. Environment configuration

`INFERENCE_URL` lets the same frontend image work under Compose, standalone Docker, or local Python by selecting the endpoint at runtime. Hardcoding a Compose-only hostname makes the image less reusable.

## 5. Internal inference port

The frontend's server process calls `http://inference:8000` across the Compose network, so inference needs no host port. Users access the published frontend and MLflow UI ports. The browser does not itself resolve `inference`; Streamlit's Python backend does.

## 6. Startup readiness

Simple `depends_on` orders processes but does not wait for a ready API. Loading a model too early raises an exception and exits inference. This implementation uses MLflow's health check, `condition: service_healthy`, and bounded model-loading retries. A new empty registry still needs a registered `champion` before inference can start successfully.

## 7. Published ports

The configuration publishes MLflow on `127.0.0.1:5000` and the frontend on `127.0.0.1:8501`. Inference exposes port 8000 only inside Docker. The execution record reports the actual `docker compose ps` result.

## 8. Selecting a new version

Refreshing the frontend still calls the API's already loaded version. Reassign `champion`, then run `docker compose restart inference` to load the new target.

## 9. Restart without rebuilding

The startup hook fetches the model each time. Code and libraries are baked into the image; registry metadata and model weights are not. No Docker rebuild is needed to change an alias target.

## 10. Persistence

`docker compose down` removes containers and network but preserves the named volume. The next `up` reuses database, registry aliases and artifacts. `down -v` deletes the named volume, so a subsequent stack has no model or alias and inference cannot load `champion`. Test the destructive case only in a separate disposable Compose project; the execution record states whether it was run.

## 11. Beyond one machine

Three inference replicas require routing/load balancing and suitable shared service discovery. Compose can scale containers on one host, but does not provide multi-host scheduling, automatic failover or distributed high availability. Surviving host failure requires an orchestrator and resilient external metadata/artifact stores—for example PostgreSQL and object storage with backups—instead of a single-host SQLite volume.
