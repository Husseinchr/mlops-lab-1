"""Apply a two-standard-error gate to a candidate registry version."""

import argparse
import math
import os

import mlflow
from mlflow import MlflowClient


def promotion_decision(candidate, champion, n):
    if n <= 0 or not all(
        math.isfinite(a) and 0 <= a <= 1 for a in (candidate, champion)
    ):
        raise ValueError("Accuracies must be in [0, 1] and n must be positive")
    margin = 2 * math.sqrt(champion * (1 - champion) / n)
    return candidate >= champion + margin, margin


def promote(client, version, name="food11"):
    candidate = client.get_model_version(name, str(version))
    champion = client.get_model_version_by_alias(name, "champion")
    if candidate.version == champion.version:
        raise ValueError("Candidate is already champion")
    candidate_run = client.get_run(candidate.run_id)
    champion_run = client.get_run(champion.run_id)
    candidate_n = int(candidate_run.data.params["test_size"])
    n = int(champion_run.data.params["test_size"])
    if (
        n != candidate_n
        or candidate_run.data.params["dataset"] != champion_run.data.params["dataset"]
    ):
        raise ValueError(
            "Candidates must use the same evaluation dataset and sample size"
        )
    if (
        candidate_run.data.params["dataset_sha256"]
        != champion_run.data.params["dataset_sha256"]
    ):
        raise ValueError("Dataset content hashes differ")
    acc_candidate = candidate_run.data.metrics["test_accuracy"]
    acc_champion = champion_run.data.metrics["test_accuracy"]
    accepted, margin = promotion_decision(acc_candidate, acc_champion, n)
    decision = "champion" if accepted else "challenger"
    client.set_registered_model_alias(name, decision, str(version))
    if accepted:
        try:
            challenger = client.get_model_version_by_alias(name, "challenger")
            if challenger.version == str(version):
                client.delete_registered_model_alias(name, "challenger")
        except mlflow.exceptions.MlflowException as error:
            if error.error_code != "RESOURCE_DOES_NOT_EXIST":
                raise
    tags = {
        "gate.decision": decision,
        "gate.candidate_accuracy": acc_candidate,
        "gate.champion_accuracy": acc_champion,
        "gate.margin": margin,
        "gate.test_size": n,
        "gate.previous_champion": champion.version,
    }
    for key, value in tags.items():
        client.set_model_version_tag(name, str(version), key, str(value))
    return tags


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, type=int)
    parser.add_argument("--name", default="food11")
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
    )
    args = parser.parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    print(promote(MlflowClient(), args.version, args.name))


if __name__ == "__main__":
    main()
