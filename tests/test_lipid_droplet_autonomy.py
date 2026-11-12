"""Regression tests for the lipid-droplet autonomous Raman workflow.

These tests deliberately stop at the software/hardware boundary.  They exercise
segmentation, target choice, adaptive revisit logic, galvo-coordinate validation,
and MDA preparation without connecting to or commanding microscope hardware.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from napari_raman_widget import dry_run


class Value:
    """Tiny stand-in for Qt spin boxes/check boxes used by dry-run helpers."""

    def __init__(self, value):
        self._value = value

    def value(self):
        return self._value


class TextSink:
    """Collect the latest status text without constructing a GUI widget."""

    def __init__(self):
        self.text = ""

    def setText(self, text):
        self.text = text


def disk(shape, center, radius):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= radius ** 2


def target(label, aim, area, radius=6.0, intensity=1.0):
    return {
        "label": label,
        "aim": np.asarray(aim, dtype=float),
        "area": float(area),
        "radius": float(radius),
        "mean_intensity": float(intensity),
        "grid": np.empty((0, 2), dtype=float),
    }


def test_watershed_finds_two_bright_droplets_and_blank_is_empty():
    """Protect the BODIPY baseline from collapsing into zero or giant masks."""
    image = np.zeros((96, 96), dtype=np.float32)
    image[disk(image.shape, (28, 28), 7)] = 1.0
    image[disk(image.shape, (68, 66), 8)] = 0.8

    labels = dry_run._watershed_masks(
        image, background_sigma=12, min_distance=4, threshold_scale=1.0
    )

    assert int(labels.max()) == 2
    assert int(dry_run._watershed_masks(np.zeros_like(image)).max()) == 0


def test_plane_to_targets_keeps_one_safe_interior_aim_and_rejects_tiny_object():
    """Protect the rule that Raman aims must lie safely inside valid droplets."""
    masks = np.zeros((80, 80), dtype=np.uint32)
    masks[disk(masks.shape, (30, 30), 9)] = 1
    masks[disk(masks.shape, (62, 62), 2)] = 2
    image = (masks > 0).astype(np.float32)
    params = {
        "density": 6,
        "margin": 4.0,
        "min_area": 30,
        "min_rel_intensity": 0.0,
        "max_area": 1000,
    }

    targets = dry_run._plane_to_targets(image, masks, params)

    assert len(targets) == 1
    chosen = targets[0]
    y, x = map(int, chosen["aim"])
    assert chosen["label"] == 1
    assert masks[y, x] == 1
    assert chosen["radius"] >= params["margin"]
    assert chosen["area"] == int(np.count_nonzero(masks == 1))


def test_adaptive_matching_is_one_to_one_and_reports_area_change():
    """Protect droplet identity and the morphology-change calculation."""
    widget = SimpleNamespace(
        _dry={
            "baseline": [
                {"z": 0, "aim": np.array([10.0, 10.0]), "area": 100.0},
                {"z": 0, "aim": np.array([50.0, 50.0]), "area": 100.0},
            ],
            "all_targets": {
                0: [
                    target(1, (11, 10), 125),
                    target(2, (50, 51), 102),
                    target(3, (90, 90), 300),
                ]
            },
        },
        dry_match_distance_input=Value(10.0),
    )

    matches = dry_run._match_adaptive_targets(widget)

    assert len(matches) == 2
    assert len({row["key"] for row in matches}) == 2
    changes = sorted(round(row["area_change_pct"], 6) for row in matches)
    assert changes == [2.0, 25.0]
    assert all(row["distance_px"] == pytest.approx(1.0) for row in matches)


def test_adaptive_priority_keeps_changing_droplet_and_stable_control(monkeypatch):
    """Protect the experimental design: responders plus internal stable controls."""
    changing = target(1, (11, 10), 125)
    stable = target(2, (50, 51), 102)
    widget = SimpleNamespace(
        _dry={
            "baseline": [
                {"z": 0, "aim": np.array([10.0, 10.0]), "area": 100.0},
                {"z": 0, "aim": np.array([50.0, 50.0]), "area": 100.0},
            ],
            "all_targets": {0: [changing, stable]},
            "adaptive_matches": [],
            "adaptive_keep": None,
        },
        dry_match_distance_input=Value(10.0),
        dry_change_threshold_input=Value(15.0),
        dry_stable_threshold_input=Value(5.0),
        dry_change_count_input=Value(5),
        dry_stable_count_input=Value(2),
        dry_adaptive_status=TextSink(),
        dry_status=TextSink(),
        status=TextSink(),
        viewer=SimpleNamespace(layers={}),
    )
    monkeypatch.setattr(dry_run, "_refresh_target_layers", lambda _widget: None)

    dry_run._adaptive_prioritize(widget)

    assert widget._dry["adaptive_keep"] == {(0, 1), (0, 2)}
    assert changing["adaptive_class"] == "changing"
    assert changing["area_change_pct"] == pytest.approx(25.0)
    assert stable["adaptive_class"] == "stable_control"
    assert stable["area_change_pct"] == pytest.approx(2.0)


@pytest.mark.parametrize(
    "volts, expected",
    [
        ((0.25, -0.75), True),
        ((1.7, 0.0), False),
        ((np.nan, 0.0), False),
    ],
)
def test_galvo_validation_checks_limits_without_writing_hardware(volts, expected):
    """Protect the dry-run boundary: coordinate validation must never move galvos."""
    hardware_writes = []
    widget = SimpleNamespace(
        _dry={"targets": {0: [target(1, (20, 30), 100)]}},
        transformer=object(),
        _pt_to_volts=lambda _point: np.asarray([volts], dtype=float),
        daq=SimpleNamespace(
            set_galvo_position=lambda value: hardware_writes.append(value)
        ),
        dry_status=TextSink(),
        status=TextSink(),
    )

    result = dry_run._validate_galvo_mapping(widget)

    assert result is expected
    assert hardware_writes == []
    if expected:
        assert len(widget._dry["galvo_preview"]) == 1
    else:
        assert widget._dry["galvo_preview"] == []


def test_prepare_raman_mda_builds_one_point_per_droplet_but_never_runs(
    monkeypatch,
):
    """Protect separation between preparing an MDA plan and starting acquisition."""
    source = SimpleNamespace(_points=SimpleNamespace(add=MagicMock()))
    fake_sequence = SimpleNamespace(stage_positions=(object(),))
    apply_prepared = MagicMock()
    acquisition_run = MagicMock()

    import cns_control.utils
    import napari_raman_widget.unified_mda as unified_mda

    monkeypatch.setattr(
        cns_control.utils, "create_point_sources", lambda *a, **k: [source]
    )
    monkeypatch.setattr(
        cns_control.utils, "_get_seq_from_napari",
        lambda *a, **k: fake_sequence,
    )
    monkeypatch.setattr(unified_mda, "apply_prepared_positions", apply_prepared)
    monkeypatch.setattr(dry_run, "_validate_galvo_mapping", lambda _widget: True)

    widget = SimpleNamespace(
        _dry={
            "targets": {
                0: [
                    target(1, (10, 20), 100),
                    target(2, (30, 40), 120),
                    target(3, (50, 60), 140),
                ]
            }
        },
        core=object(),
        main_window=object(),
        viewer=SimpleNamespace(layers={}),
        dry_pos_input=Value(0),
        _make_point_transformer=lambda size, count: ("point", size, count),
        _reset_grid_regions=MagicMock(),
        _refresh_cell_layer_controls=MagicMock(),
        _activate_selected_cell_layer=MagicMock(),
        _acquisition_controller=SimpleNamespace(run=acquisition_run),
        dry_status=TextSink(),
        status=TextSink(),
    )

    dry_run._prepare_raman_mda(widget)

    assert source._points.add.call_count == 3
    assert widget.selection_results["selection_type"] == "droplet"
    assert widget.selection_results["pre_acq"] is False
    apply_prepared.assert_called_once_with(widget)
    acquisition_run.assert_not_called()
