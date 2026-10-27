import shutil

from PIL import Image

from src.food11.data import CLASSES, SPLITS
from src.food11.validate_data import validate_dataset


def test_contract_detects_leakage_corruption_and_missing_class(tmp_path):
    for s, split in enumerate(SPLITS):
        for c, category in enumerate(CLASSES):
            folder = tmp_path / split / category
            folder.mkdir(parents=True)
            Image.new("RGB", (128, 128), (s * 70, c * 20, 10)).save(
                folder / "image.png"
            )
    checks, _, _ = validate_dataset(tmp_path, min_per_class=1)
    assert all(checks.values())
    shutil.copy2(
        tmp_path / "training/Bread/image.png", tmp_path / "evaluation/Bread/leak.png"
    )
    checks, _, _ = validate_dataset(tmp_path, min_per_class=1)
    assert not checks["duplicates"]
    (tmp_path / "evaluation/Bread/leak.png").unlink()
    (tmp_path / "validation/Egg/image.png").write_bytes(b"corrupted")
    checks, _, _ = validate_dataset(tmp_path, min_per_class=1)
    assert not checks["images"]
    (tmp_path / "validation/Egg/image.png").unlink()
    (tmp_path / "validation/Egg").rmdir()
    checks, _, _ = validate_dataset(tmp_path, min_per_class=1)
    assert not checks["folders"] and not checks["minimum"]
