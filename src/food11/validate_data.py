"""Validate the processed Food-11 data contract before training."""

import argparse
import hashlib
from pathlib import Path

from PIL import Image

try:
    from .data import CLASSES, SPLITS
except ImportError:
    from data import CLASSES, SPLITS


def validate_dataset(root: Path, min_per_class=50, tau=0.05):
    if min_per_class < 1 or not 0 <= tau <= 1:
        raise ValueError("min-per-class must be positive; tau must be in [0, 1]")
    checks = dict.fromkeys(
        ["folders", "images", "minimum", "duplicates", "balance"], True
    )
    counts = {}
    seen = {}
    details = []
    for split in SPLITS:
        directory = root / split
        folders = (
            {p.name for p in directory.iterdir() if p.is_dir()}
            if directory.is_dir()
            else set()
        )
        checks["folders"] &= folders == set(CLASSES)
        counts[split] = {}
        for category in CLASSES:
            files = sorted((directory / category).glob("*"))
            files = [p for p in files if p.is_file()]
            counts[split][category] = len(files)
            checks["minimum"] &= len(files) >= min_per_class
            for path in files:
                try:
                    with Image.open(path) as img:
                        img.load()
                        checks["images"] &= img.mode == "RGB" and img.size == (128, 128)
                    digest = hashlib.md5(path.read_bytes()).hexdigest()
                    if digest in seen and seen[digest][0] != split:
                        checks["duplicates"] = False
                        details.append(f"Duplicate: {seen[digest][1]} == {path}")
                    else:
                        seen[digest] = (split, path)
                except (OSError, ValueError) as error:
                    checks["images"] = False
                    details.append(f"Invalid image: {path}: {error}")
    for category in CLASSES:
        totals = {s: sum(counts[s].values()) for s in ("training", "validation")}
        if not all(totals.values()):
            checks["balance"] = False
        else:
            gap = abs(
                counts["training"][category] / totals["training"]
                - counts["validation"][category] / totals["validation"]
            )
            checks["balance"] &= gap <= tau
    return checks, counts, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--min-per-class", type=int, default=50)
    parser.add_argument("--tau", type=float, default=0.05)
    args = parser.parse_args()
    checks, counts, details = validate_dataset(
        args.dataset, args.min_per_class, args.tau
    )
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    for split, categories in counts.items():
        print(f"{split}: {sum(categories.values())} images")
    for detail in details[:10]:
        print(detail)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
