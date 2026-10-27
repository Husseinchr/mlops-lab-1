"""Register a run's probability adapter and optionally assign an alias."""

import argparse
import os

import mlflow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--alias")
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
    )
    args = parser.parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    version = mlflow.register_model(f"runs:/{args.run_id}/classifier", "food11")
    if args.alias:
        mlflow.MlflowClient().set_registered_model_alias(
            "food11", args.alias, version.version
        )
    print(f"VERSION={version.version} RUN_ID={args.run_id} ALIAS={args.alias}")


if __name__ == "__main__":
    main()
