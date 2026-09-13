"""Prepare Food-11 for class-folder image classification workflows."""

from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps


IMAGE_SIZE = (128, 128)
MINI_IMAGES_PER_CLASS = 100
SPLITS = ("training", "evaluation", "validation")
CLASS_NAMES = {
    "0": "Bread",
    "1": "Dairy product",
    "2": "Dessert",
    "3": "Egg",
    "4": "Fried food",
    "5": "Meat",
    "6": "Noodles-Pasta",
    "7": "Rice",
    "8": "Seafood",
    "9": "Soup",
    "10": "Vegetable-Fruit",
}

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPOSITORY_ROOT / "data"
RAW_ROOT = DATA_ROOT / "food11_raw"
PROCESSED_ROOT = DATA_ROOT / "food11_processed"
MINI_ROOT = DATA_ROOT / "food11_processed_mini"


def class_name_from_filename(image_path: Path) -> str:
    """Return the class directory associated with a Food-11 filename."""
    label = image_path.stem.split("_", maxsplit=1)[0]
    try:
        return CLASS_NAMES[label]
    except KeyError as error:
        raise ValueError(f"Unknown class label in {image_path.name!r}") from error


def resize_image(source: Path, destination: Path) -> None:
    """Resize one image to 128x128 and save it as an RGB JPEG."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        image.save(destination, format="JPEG", quality=90, optimize=True)


def reset_output_directories() -> None:
    """Remove generated outputs so reruns cannot retain stale files."""
    for output_root in (PROCESSED_ROOT, MINI_ROOT):
        if output_root.exists():
            shutil.rmtree(output_root)


def prepare_split(split: str) -> tuple[int, int]:
    """Prepare one split and return full and mini image counts."""
    raw_split = RAW_ROOT / split
    if not raw_split.is_dir():
        raise FileNotFoundError(f"Missing raw-data split: {raw_split}")

    mini_counts: Counter[str] = Counter()
    processed_count = 0
    mini_count = 0

    for source in sorted(raw_split.glob("*.jpg")):
        class_name = class_name_from_filename(source)
        processed_destination = PROCESSED_ROOT / split / class_name / source.name
        resize_image(source, processed_destination)
        processed_count += 1

        if mini_counts[class_name] < MINI_IMAGES_PER_CLASS:
            mini_destination = MINI_ROOT / split / class_name / source.name
            mini_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(processed_destination, mini_destination)
            mini_counts[class_name] += 1
            mini_count += 1

    missing_classes = set(CLASS_NAMES.values()) - set(mini_counts)
    if missing_classes:
        missing = ", ".join(sorted(missing_classes))
        raise ValueError(f"Split {split!r} is missing classes: {missing}")

    return processed_count, mini_count


def main() -> None:
    """Build the full and mini processed Food-11 datasets."""
    reset_output_directories()
    total_processed = 0
    total_mini = 0

    for split in SPLITS:
        processed_count, mini_count = prepare_split(split)
        total_processed += processed_count
        total_mini += mini_count
        print(f"{split}: {processed_count} processed, {mini_count} mini")

    print(f"Done: {total_processed} processed, {total_mini} mini")


if __name__ == "__main__":
    main()
