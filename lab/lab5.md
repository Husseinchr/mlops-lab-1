# Lab 5 — CI/CD and Model Promotion

## 1. Isolated unit tests

A real model load can fail because MLflow is down, credentials expire, the alias is absent, artifacts are unreachable, or weights are too large for runner memory. These do not necessarily indicate incorrect API code. `SKIP_MODEL` and an injected fake make unit tests deterministic and independent of those services.

## 2. What the fake proves

The API test exercises multipart upload, image decoding, NCHW preprocessing, canonical class lookup and JSON confidence serialization. It does not measure real classification accuracy, registry access, artifact retrieval or the semantic correctness of trained weights. Invalid-image and malformed-output tests cover error behavior separately.

## 3. Duplicate leakage

The cross-split content-hash check detects the intentionally copied image. A duplicate can inflate evaluation accuracy silently; a corrupt image usually causes a visible failure. The supplied dataset also contained real cross-split duplicates, so generated data was deduplicated before training while preserving the original raw data.

## 4. Where to validate full data

Run the validator before every training execution, after obtaining the DVC snapshot. Also run it on a data-update workstation or a dedicated self-hosted CI runner that can pull the data. Hosted unit tests validate the checker using a small synthetic dataset, including a deliberately introduced duplicate; they do not claim to validate unseen DVC data.

## 5. Job dependencies

With `needs: test`, a failing test job prevents smoke from running. Without it, the jobs can run concurrently and consume build resources despite failing unit tests. Publishing similarly depends on the smoke job.

## 6. Cold and cached run timing

Measured GitHub Actions timings belong in the execution record. setup-uv caches downloaded dependency artifacts; subsequent `uv sync --frozen` installations can reuse them. A cache does not remove the need to check out code, create the environment and run tests. Docker publishing separately uses a GitHub Actions build cache.

## 7. SHA tags

Suppose laptop A pulls `latest` on Monday and server B pulls it after Tuesday's release. Both commands name `latest`, but cached/running images may contain different code. A commit-SHA tag associates the release with its source revision; a digest pins exact image bytes. The model alias is a separate version decision.

## 8. Token permissions

The test and smoke jobs need only repository read access. Only the publishing job receives `packages: write`. A broadly privileged compromised step could rewrite code or releases, alter repository settings if permitted, or publish malicious packages. Explicit least privilege limits the token's reach.

## 9. Branch protection

A required-check rule blocks merging a pull request until test and smoke pass; requiring a pull request prevents ordinary direct pushes. Enforcement is centralized and repeatable rather than depending on each developer remembering to run local tests. Administrators may bypass rules if the rule permits it. The execution record states whether the rule and deliberately failing PR were actually verified.

## 10. Quality gate

The candidate becomes champion only if `candidate >= champion + 2*sqrt(p*(1-p)/n)`. Otherwise it receives `challenger`. Tags record both accuracies, the previous champion version, sample size, margin and decision. The script verifies matching dataset hashes and test sizes before comparing. Exact candidate results are recorded after execution in `lab-answers.md`.

This is the prescribed lecture heuristic, not a general significance test for two classifiers evaluated on the same examples; a paired test would use their per-example disagreement.

## 11. Automatic promotion

One option is a post-training step on a self-hosted runner or orchestrated job that can reach the local tracking server. Another is a hosted MLflow server such as DagsHub, with credentials injected into the training/promotion job. Both must preserve the quality gate and log its result.

## 12. Pipeline

```mermaid
flowchart TD
    A[Human: edit code and git push] --> B[Automatic: lint and unit tests]
    B --> C[Automatic: build API and model-free smoke test]
    C --> D{Push to main?}
    D -->|yes| E[Automatic: publish image tagged with Git SHA]
    E --> F[Human: deploy exact image SHA or digest]
    F --> G[Container startup loads food11 champion]
    H[Human: choose data and training settings] --> I[Training: validate data and log MLflow run]
    I --> J[Register candidate version]
    J --> K[Promotion script: compare accuracy with gate]
    K --> L{Gate passes?}
    L -->|yes| M[Move champion alias]
    L -->|no| N[Assign challenger; keep champion]
    M --> O[Human: restart inference]
    O --> G
```

The Git commit decides the application code version; the registry version selected by `champion` decides the model weights. Training, candidate registration, promotion invocation and deployment/restart remain explicit local operations in this lab; GitHub automatically tests and publishes the application image.
