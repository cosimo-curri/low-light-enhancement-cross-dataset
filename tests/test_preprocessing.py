from __future__ import annotations

import numpy as np
from PIL import Image

from scripts.preprocess_offline import compute_image_statistics, process_image
from src.low_light_enhancement.framework.io import read_rgb_image


def test_resize_preserves_pixels_using_pillow_lanczos(tmp_path) -> None:
    source = tmp_path / "input.png"
    output = tmp_path / "nested/output.png"
    pixels = np.arange(3 * 11 * 19, dtype=np.uint8).reshape(11, 19, 3)
    Image.fromarray(pixels, mode="RGB").save(source)

    stats = process_image(
        source_path=source,
        output_path=output,
        image_size=(32, 32),
        output_format="png",
        dark_threshold=0.10,
    )

    expected = read_rgb_image(source).resize((32, 32), Image.Resampling.LANCZOS)
    with Image.open(output) as actual:
        assert actual.mode == "RGB"
        assert np.array_equal(np.asarray(actual), np.asarray(expected))
    assert stats == compute_image_statistics(read_rgb_image(source), 0.10)
    assert stats["width"] == 19.0
    assert stats["height"] == 11.0


def test_input_statistics_are_from_unresized_image() -> None:
    original = Image.fromarray(np.full((7, 9, 3), 5, dtype=np.uint8), mode="RGB")
    stats = compute_image_statistics(original, 0.10)
    assert stats["dark_pixel_ratio"] == 1.0
    assert stats["mean_luminance"] == np.float32(5 / 255) or abs(stats["mean_luminance"] - 5/255) < 1e-6
