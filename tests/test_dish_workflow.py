import numpy as np
import pytest

from napari_raman_widget.dish_workflow import (
    export_targets_csv, pixel_to_stage, segment_tiles, stage_targets)


def target(row, col, radius=3.0):
    return {"label": 1, "aim": (row, col), "radius": radius,
            "grid": np.array([[row, col]], float)}


def test_center_pixel_maps_to_tile_position():
    xy = pixel_to_stage([(49.5, 49.5)], (1000, 2000), (100, 100), 0.5)
    assert xy[0] == pytest.approx([1000, 2000])


def test_offset_and_flipped_axes():
    xy = pixel_to_stage([(49.5, 59.5)], (0, 0), (100, 100), 2.0)
    assert xy[0] == pytest.approx([20, 0])
    xy = pixel_to_stage([(49.5, 59.5)], (0, 0), (100, 100), 2.0, axes=((-1, 0), (0, 1)))
    assert xy[0] == pytest.approx([-20, 0])


def test_overlap_duplicate_is_kept_once_from_the_less_edge_tile():
    # Same droplet at stage (50, 0): near the right edge of tile 0, mid-way in tile 1.
    tiles = [(0, 0), (100, 0)]
    shape = (101, 101)
    records = [
        {"tile": 0, "tile_xy": tiles[0], "shape": shape, "targets": [target(50, 100)]},
        {"tile": 1, "tile_xy": tiles[1], "shape": shape, "targets": [target(50, 0)]},
    ]
    kept = stage_targets(records, pixel_um=1.0)
    assert len(kept) == 1
    assert kept[0]["xy"] == pytest.approx([50, 0])


def test_distinct_droplets_survive_and_csv_written(tmp_path):
    records = segment_tiles(
        [(0, 0)], [np.zeros((101, 101))],
        lambda img: [target(10, 10), target(80, 80)])
    kept = stage_targets(records, pixel_um=1.0)
    assert len(kept) == 2
    path = tmp_path / "t.csv"
    export_targets_csv(path, kept)
    assert len(path.read_text().splitlines()) == 3


def test_segment_tiles_needs_matching_lengths():
    with pytest.raises(ValueError):
        segment_tiles([(0, 0)], [], lambda i: [])


def test_split_into_tiles_covers_image_with_overlap():
    from napari_raman_widget.dish_panel import split_into_tiles
    img = np.arange(100 * 100, dtype=float).reshape(100, 100)
    tiles, crops = split_into_tiles(img, 60, 0.5, 2.0)
    assert all(c.shape == (60, 60) for c in crops)
    assert len(tiles) == len(crops) == 9
    # Last tile reaches the far edge, so no pixels are skipped.
    assert crops[-1][-1, -1] == img[-1, -1]
    assert tiles[0] == pytest.approx((29.5 * 2, 29.5 * 2))
