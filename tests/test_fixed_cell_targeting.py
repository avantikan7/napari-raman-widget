"""Hardware-free tests for fixed-cell Z-stack LD targeting."""
import numpy as np
import pytest
from scipy import ndimage as ndi

from napari_raman_widget.fixed_cell import (
    PixelScale,
    area_px_to_diameter_um,
    assign_targets_to_cells,
    cellpose_volume_kwargs,
    diameter_um_to_area_px,
    focused_droplet_targets,
)


def disk(shape, center, radius):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= radius ** 2


def test_one_micron_physical_limit_at_koseki_scale():
    scale = PixelScale(0.325, 0.325)
    area = diameter_um_to_area_px(1.0, scale)
    assert area == pytest.approx(7.435, rel=0.01)
    assert area_px_to_diameter_um(area, scale) == pytest.approx(1.0)


def test_cellpose_modes_expose_stitch_and_25d_parameters():
    scale = PixelScale(0.325, 0.325)
    stitched = cellpose_volume_kwargs(
        "stitch", z_step_um=0.5, scale=scale,
        cellprob=1.0, flow=0.4, stitch=0.55,
    )
    assert stitched["do_3D"] is False
    assert stitched["stitch_threshold"] == pytest.approx(0.55)
    assert stitched["cellprob_threshold"] == pytest.approx(1.0)

    volume = cellpose_volume_kwargs(
        "3d", z_step_um=0.5, scale=scale,
        cellprob=1.0, flow=0.4, stitch=0.55,
    )
    assert volume["do_3D"] is True
    assert volume["anisotropy"] == pytest.approx(0.5 / 0.325)
    assert volume["stitch_threshold"] == 0.0


def test_best_focus_plane_and_physical_filtering():
    shape = (5, 64, 64)
    labels = np.zeros(shape, np.uint32)
    mask = disk(shape[1:], (32, 32), 5)
    labels[:, mask] = 1

    sharp = np.zeros(shape[1:], np.float32)
    sharp[mask] = 1.0
    blurred = ndi.gaussian_filter(sharp, 3.0)
    stack = np.stack([blurred, blurred, sharp, blurred, blurred])

    focused, targets = focused_droplet_targets(
        stack,
        labels,
        scale=PixelScale(0.325, 0.325),
        min_diameter_um=1.0,
        max_diameter_um=10.0,
        min_focus_confidence=0.01,
        min_z_span=1,
        max_z_span=0,
        reject_edge_focus=True,
        grid_spacing_um=1.0,
        grid_margin_um=0.5,
    )
    assert len(targets) == 1
    assert targets[0]["z"] == 2
    assert targets[0]["diameter_um"] > 1.0
    assert np.count_nonzero(focused[2]) == np.count_nonzero(mask)
    assert np.count_nonzero(focused[[0, 1, 3, 4]]) == 0

    _focused, rejected = focused_droplet_targets(
        stack,
        labels,
        scale=PixelScale(0.325, 0.325),
        min_diameter_um=5.0,
        max_diameter_um=10.0,
        min_focus_confidence=0.01,
        min_z_span=1,
        max_z_span=0,
        reject_edge_focus=True,
        grid_spacing_um=1.0,
        grid_margin_um=0.5,
    )
    assert rejected == []


def test_assign_targets_to_parent_cells_and_limit_per_cell():
    cells = np.zeros((50, 50), np.uint32)
    cells[:, :25] = 1
    cells[:, 25:] = 2
    targets = [
        {"id": 1, "aim": (10, 10), "focus_confidence": 0.9, "radius_px": 4},
        {"id": 2, "aim": (15, 15), "focus_confidence": 0.5, "radius_px": 3},
        {"id": 3, "aim": (20, 35), "focus_confidence": 0.8, "radius_px": 4},
    ]
    assign_targets_to_cells(targets, cells, require_cell=True, max_per_cell=1)
    assert targets[0]["cell_id"] == 1 and targets[0]["selected"]
    assert targets[1]["cell_id"] == 1 and not targets[1]["selected"]
    assert targets[2]["cell_id"] == 2 and targets[2]["selected"]
