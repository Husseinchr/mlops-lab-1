import pytest
from PIL import Image

from src.food11.data import CLASSES, category_from_filename, resize_image


@pytest.mark.parametrize(
    "filename,category",
    [("0_12.jpg", "Bread"), ("10_3.jpg", "Vegetable-Fruit"), ("3_120.jpg", "Egg")],
)
def test_category(filename, category):
    assert category_from_filename(filename) == category


@pytest.mark.parametrize("filename", ["11_0.jpg", "-1_0.jpg", "garbage.jpg"])
def test_invalid_category(filename):
    with pytest.raises(ValueError):
        category_from_filename(filename)


def test_resize():
    image = Image.new("L", (512, 384))
    result = resize_image(image)
    assert result.size == (128, 128)
    assert result.mode == "RGB"
    assert image.size == (512, 384)


def test_classes():
    assert len(CLASSES) == len(set(CLASSES)) == 11
