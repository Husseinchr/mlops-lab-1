"""Train a pretrained ResNet18 and log reproducible MLflow experiments."""

import argparse
import hashlib
import os
import random
import sys
import tempfile
from pathlib import Path

import mlflow.pytorch
import numpy as np
import torch
from mlflow.models import infer_signature
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights, resnet18

import mlflow
from food11.data import CLASSES, DATA_ROOT
from food11.model import ProbabilityModel, image_transform
from food11.validate_data import validate_dataset


def evaluate(model, loader, criterion):
    model.eval()
    loss, correct, total = 0.0, 0, 0
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images)
            loss += criterion(logits, labels).item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)
    return loss / total, correct / total


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["processed", "mini"], default="mini")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.lr <= 0 or args.batch_size < 1 or args.threads < 1:
        parser.error("epochs, lr, batch-size and threads must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    root = DATA_ROOT / (
        "food11_processed_mini" if args.dataset == "mini" else "food11_processed"
    )
    print(f"Validating {root}", flush=True)
    checks, _, details = validate_dataset(root)
    if not all(checks.values()):
        raise ValueError(f"Dataset contract failed: {checks}; {details[:3]}")
    datasets = {
        s: ImageFolder(root / s, transform=image_transform())
        for s in ("training", "validation", "evaluation")
    }
    # ImageFolder sorts alphabetically; map labels back to Food-11 numeric order.
    for dataset in datasets.values():
        mapping = [CLASSES.index(name) for name in dataset.classes]
        dataset.target_transform = lambda label, mapping=mapping: mapping[label]
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        s: DataLoader(
            d,
            batch_size=args.batch_size,
            shuffle=s == "training",
            num_workers=0,
            generator=generator,
        )
        for s, d in datasets.items()
    }
    torch.hub.set_dir(str(DATA_ROOT.parent / ".verification" / "torch" / "hub"))
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    # Freeze the backbone for a practical CPU transfer-learning experiment.
    model.requires_grad_(False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.lr)
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.jpg")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    dataset_hash = digest.hexdigest()
    # Deterministic frozen features are reusable across hyperparameter runs.
    # Cache identity includes every image byte, preprocessing and torch version.
    cache = (
        DATA_ROOT.parent
        / ".verification"
        / f"features-{dataset_hash}-{torch.__version__}-v1.pt"
    )
    if cache.exists():
        print("Using cached frozen ResNet18 features", flush=True)
        features = torch.load(cache, weights_only=True)
    else:
        head = model.fc
        model.fc = nn.Identity()
        model.eval()
        features = {}
        with torch.inference_mode():
            for split, dataset in datasets.items():
                print(f"Extracting {split}: {len(dataset)} images", flush=True)
                batches, targets = [], []
                for images, labels in DataLoader(dataset, batch_size=args.batch_size):
                    batches.append(model(images).clone())
                    targets.append(labels.clone())
                features[split] = (torch.cat(batches), torch.cat(targets))
        model.fc = head
        cache.parent.mkdir(exist_ok=True)
        torch.save(features, cache)
    loaders = {
        s: DataLoader(
            TensorDataset(*values),
            batch_size=args.batch_size,
            shuffle=s == "training",
            generator=generator,
        )
        for s, values in features.items()
    }
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("food11")
    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "dataset": args.dataset,
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "architecture": "resnet18",
                "backbone_frozen": True,
                "test_size": len(datasets["evaluation"]),
            }
        )
        mlflow.log_param("dataset_sha256", dataset_hash)
        mlflow.log_dict(
            {"classes": CLASSES, "class_to_idx": {c: i for i, c in enumerate(CLASSES)}},
            "classes.json",
        )
        for epoch in range(1, args.epochs + 1):
            # eval keeps pretrained batch-normalization statistics fixed.
            model.eval()
            model.fc.train()
            total_loss = 0.0
            for images, labels in loaders["training"]:
                optimizer.zero_grad()
                logits = model.fc(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(labels)
            val_loss, val_accuracy = evaluate(
                model.fc, loaders["validation"], criterion
            )
            metrics = {
                "train_loss": total_loss / len(datasets["training"]),
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
            mlflow.log_metrics(metrics, step=epoch)
            print(f"epoch {epoch}: {metrics}", flush=True)
        _, test_accuracy = evaluate(model.fc, loaders["evaluation"], criterion)
        mlflow.log_metric("test_accuracy", test_accuracy)
        model.eval().cpu()
        # Keep the required PyTorch artifact and register a pyfunc adapter that
        # converts network logits to probabilities in canonical class order.
        example = np.zeros((1, 3, 128, 128), dtype=np.float32)
        mlflow.pytorch.log_model(
            model, name="model", serialization_format="pickle", input_example=example
        )
        with tempfile.TemporaryDirectory() as directory:
            network_path = str(Path(directory) / "network")
            mlflow.pytorch.save_model(
                model, network_path, serialization_format="pickle"
            )
            mlflow.pyfunc.log_model(
                name="classifier",
                python_model=ProbabilityModel(),
                artifacts={"network": network_path},
                code_paths=[str(Path(__file__).parent)],
                signature=infer_signature(example, np.zeros((1, 11), dtype=np.float32)),
                input_example=example,
                pip_requirements=[
                    f"torch=={torch.__version__.split('+')[0]}",
                    f"mlflow=={mlflow.__version__}",
                    "numpy",
                    "pillow",
                    "torchvision",
                ],
            )
        print(f"RUN_ID={run.info.run_id} TEST_ACCURACY={test_accuracy:.6f}", flush=True)


if __name__ == "__main__":
    main()
