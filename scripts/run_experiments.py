"""Run Lab 2's comparison, register the best run, and exercise Lab 5's gate."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import mlflow
from food11.promote import promote
from mlflow import MlflowClient


def train(lr, batch):
    subprocess.run(
        [
            sys.executable,
            "src/food11/train.py",
            "--dataset",
            "mini",
            "--epochs",
            "5",
            "--lr",
            str(lr),
            "--batch-size",
            str(batch),
        ],
        check=True,
    )


def save_report(report):
    destination = Path(".verification/experiments.json")
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["compare", "promote", "all"], default="all")
    args = parser.parse_args()
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    client = MlflowClient()
    if args.phase in ("compare", "all"):
        for lr, batch in [(0.01, 32), (0.001, 32), (0.0001, 32), (0.001, 64)]:
            train(lr, batch)
    experiment = client.get_experiment_by_name("food11")
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.val_accuracy DESC"],
        max_results=100,
    )
    if args.phase == "promote":
        report = json.loads(
            Path(".verification/experiments.json").read_text(encoding="utf-8")
        )
        best = client.get_run(report["best_run"])
    else:
        best = runs[0]
        champion = mlflow.register_model(
            f"runs:/{best.info.run_id}/classifier", "food11"
        )
        client.set_registered_model_alias("food11", "champion", champion.version)
        report = {
            "runs": [
                {
                    "id": r.info.run_id,
                    "params": r.data.params,
                    "metrics": r.data.metrics,
                }
                for r in runs
            ],
            "best_run": best.info.run_id,
            "initial_champion_version": champion.version,
            "candidates": [],
        }
        save_report(report)
    if args.phase == "compare":
        print(json.dumps(report, indent=2))
        return
    for lr in [0.000001, float(best.data.params["lr"])]:
        before = {r.info.run_id for r in client.search_runs([experiment.experiment_id])}
        train(lr, int(best.data.params["batch_size"]))
        new_run = next(
            r
            for r in client.search_runs([experiment.experiment_id])
            if r.info.run_id not in before and r.info.status == "FINISHED"
        )
        version = mlflow.register_model(
            f"runs:/{new_run.info.run_id}/classifier", "food11"
        )
        decision = promote(client, version.version)
        report["candidates"].append(
            {"run_id": new_run.info.run_id, "version": version.version, **decision}
        )
        save_report(report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
