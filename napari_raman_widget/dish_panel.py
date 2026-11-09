"""Dish workflow panel: tile a 12 mm well, segment each tile, review, export.

Hardware-free. Pass 1 segments tile images (a folder of tile TIFFs in tile
order, or the selected image layer split into overlapping tiles as a
simulation). Pass 2 maps droplets to stage micrometers, removes duplicates in
tile overlaps and shows them in a napari layer for review.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from qtpy.QtWidgets import (
    QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QPushButton, QSpinBox,
    QVBoxLayout,
)

from .dish_workflow import export_targets_csv, segment_tiles, stage_targets
from .ui_helpers import make_collapsible

REVIEW, WELL = "dish droplets (stage um)", "dish well outline"


def split_into_tiles(image, tile_px, overlap, pixel_um):
    """Overlapping square crops of ``image`` with stage-centre positions (um)."""
    h, w = image.shape[-2:]
    tile_px = min(int(tile_px), h, w)
    step = max(1, int(round(tile_px * (1 - overlap))))

    def starts(n):
        s = list(range(0, max(n - tile_px, 0) + 1, step))
        return s if s[-1] + tile_px >= n else s + [n - tile_px]

    tiles, crops = [], []
    for r in starts(h):
        for c in starts(w):
            crops.append(image[..., r:r + tile_px, c:c + tile_px])
            tiles.append(((c + (tile_px - 1) / 2) * pixel_um,
                          (r + (tile_px - 1) / 2) * pixel_um))
    return tiles, crops


def build_dish_controls(widget):
    box = make_collapsible("Dish workflow: tile, segment, review (no hardware)", expanded=False)
    lay = QVBoxLayout()
    lay.addWidget(QLabel(
        "Pass 1: segment every tile with the Dry-run settings and Cellpose model.\n"
        "Pass 2: droplets in stage um, overlaps merged. Review, then export."))

    def row(label, w):
        r = QHBoxLayout(); r.addWidget(QLabel(label)); r.addWidget(w); lay.addLayout(r)

    widget.dish_load_btn = QPushButton("Load dish plan (JSON)...")
    lay.addWidget(widget.dish_load_btn)
    widget.dish_plan_label = QLabel("No plan: using 12 mm well centred at (0, 0).")
    widget.dish_plan_label.setWordWrap(True)
    lay.addWidget(widget.dish_plan_label)
    widget.dish_pixel_input = QDoubleSpinBox()
    widget.dish_pixel_input.setDecimals(4); widget.dish_pixel_input.setRange(0.001, 100)
    widget.dish_pixel_input.setValue(0.1263)
    widget.dish_pixel_input.setSuffix(" um/px")
    row("Pixel size:", widget.dish_pixel_input)
    widget.dish_tile_input = QSpinBox()
    widget.dish_tile_input.setRange(64, 8192); widget.dish_tile_input.setValue(640)
    widget.dish_tile_input.setToolTip("Simulation only: tile edge when splitting an image layer.")
    row("Simulated tile (px):", widget.dish_tile_input)
    widget.dish_overlap_input = QDoubleSpinBox()
    widget.dish_overlap_input.setRange(0.0, 0.9); widget.dish_overlap_input.setSingleStep(0.05)
    widget.dish_overlap_input.setValue(0.4)
    row("Tile overlap:", widget.dish_overlap_input)
    widget.dish_layer_btn = QPushButton("Segment selected image layer (simulated tiles)")
    widget.dish_folder_btn = QPushButton("Segment folder of tile TIFFs...")
    widget.dish_folder_btn.setToolTip(
        "One 2D TIFF per tile (or Z-C-Y-X, using the Dry-run channel and the\n"
        "brightest plane), sorted by name in the plan's serpentine tile order.")
    lay.addWidget(widget.dish_layer_btn); lay.addWidget(widget.dish_folder_btn)
    widget.dish_export_btn = QPushButton("Export reviewed targets (CSV)")
    lay.addWidget(widget.dish_export_btn)
    widget.dish_status = QLabel("")
    widget.dish_status.setWordWrap(True)
    lay.addWidget(widget.dish_status)
    box.setLayout(lay)

    widget._dish = {"plan": None, "targets": []}
    widget.dish_load_btn.clicked.connect(lambda _=False: _load_plan(widget))
    widget.dish_layer_btn.clicked.connect(lambda _=False: _from_layer(widget))
    widget.dish_folder_btn.clicked.connect(lambda _=False: _from_folder(widget))
    widget.dish_export_btn.clicked.connect(lambda _=False: _export(widget))
    return box


def _say(widget, text):
    widget.dish_status.setText(text)


def well_geometry(widget):
    """(center_xy_um, radius_um) from the loaded plan, else a 12 mm default."""
    plan = widget._dish["plan"]
    if plan is None:
        return (0.0, 0.0), 6000.0
    p = plan.all_well_positions[0]
    return (float(p.x), float(p.y)), float(plan.plate.well_size[0]) * 500


def _load_plan(widget, path=None):
    from useq import WellPlatePlan
    from .wellplate_calibration import plate_plan_model

    if not isinstance(path, (str, Path)):
        path, _ = QFileDialog.getOpenFileName(widget, "Load dish plan", "", "JSON (*.json)")
    if not path:
        return
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        plan = plate_plan_model(data).model_validate(data)
        if not isinstance(plan, WellPlatePlan) or len(plan.all_well_positions) != 1:
            raise ValueError("A dish plan has exactly one selected well")
    except Exception as exc:
        _say(widget, f"could not load plan: {exc}")
        return
    widget._dish["plan"] = plan
    (cx, cy), r = well_geometry(widget)
    widget.dish_plan_label.setText(f"{Path(path).name}: centre ({cx:.0f}, {cy:.0f}) um, radius {r:.0f} um")


def _segmenter(widget):
    """Build segment_fn from the Dry-run panel's settings (loads the model once)."""
    from .dry_run import _plane_to_targets
    from .widget import _cellpose_diameter
    from raman_mda_engine.aiming._cellpose import eval_cellpose_model, new_cellpose_model

    model_name = widget._selected_cellpose_model(widget.sel_cellpose_combo)
    diameter = _cellpose_diameter(widget.dry_diameter_input.value())
    flow = float(widget.dry_flow_input.value())
    params = dict(density=int(widget.dry_density_input.value()),
                  margin=float(widget.dry_margin_input.value()),
                  min_rel_intensity=float(widget.dry_rel_input.value()),
                  max_area=int(widget.dry_maxarea_input.value()) or None)
    model = new_cellpose_model(model_name)

    def segment(plane):
        plane = np.asarray(plane, np.float32)
        lo, hi = np.percentile(plane, [1, 99.8])
        if hi <= lo:
            return []
        norm = np.clip((plane - lo) / (hi - lo), 0, 1)
        masks = eval_cellpose_model(model, norm, diameter=diameter, flow_threshold=flow,
                                    cellprob_threshold=-2, normalize=False)
        masks = masks[0] if isinstance(masks, tuple) else masks
        return _plane_to_targets(plane, masks, params)
    return segment


def _from_layer(widget):
    layer = widget.viewer.layers.selection.active
    if layer is None or not hasattr(layer, "data") or np.ndim(layer.data) not in (2, 3):
        _say(widget, "select a 2D or Z-Y-X image layer")
        return
    data = np.asarray(layer.data)
    plane = data if data.ndim == 2 else data[int(widget.viewer.dims.current_step[0])]
    tiles, crops = split_into_tiles(plane, widget.dish_tile_input.value(),
                                    widget.dish_overlap_input.value(),
                                    widget.dish_pixel_input.value())
    _run(widget, tiles, crops)


def _from_folder(widget):
    import tifffile
    from .dish_workflow import tile_centers

    folder = QFileDialog.getExistingDirectory(widget, "Folder of tile TIFFs")
    if not folder:
        return
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in (".tif", ".tiff"))
    if not files:
        _say(widget, "no TIFF files in that folder")
        return
    images = []
    channel = int(widget.dry_channel_input.value())
    for f in files:
        a = tifffile.imread(f)
        if a.ndim == 4:                      # Z, C, Y, X
            a = a[:, min(channel, a.shape[1] - 1)]
        if a.ndim == 3:                      # brightest plane
            a = a[int(np.argmax([np.percentile(p, 99.5) for p in a]))]
        if a.ndim != 2:
            _say(widget, f"{f.name}: need a 2D, Z-Y-X or Z-C-Y-X image")
            return
        images.append(a)
    h, w = images[0].shape
    if any(i.shape != (h, w) for i in images):
        _say(widget, "all tile images must have the same size")
        return
    px = widget.dish_pixel_input.value()
    center, radius = well_geometry(widget)
    tiles = tile_centers(center, radius, (w * px, h * px), widget.dish_overlap_input.value())
    if len(tiles) != len(images):
        _say(widget, f"{len(images)} images but the plan needs {len(tiles)} tiles "
                     f"for a {w * px:.0f} um field; adjust overlap/pixel size or the folder")
        return
    _run(widget, tiles, images)


def _run(widget, tiles, images):
    from napari.qt.threading import thread_worker

    try:
        segment = _segmenter(widget)
    except (FileNotFoundError, ValueError) as exc:
        _say(widget, str(exc))
        return
    px = widget.dish_pixel_input.value()
    buttons = (widget.dish_layer_btn, widget.dish_folder_btn)
    for b in buttons:
        b.setEnabled(False)

    @thread_worker
    def work():
        records = []
        for i, (xy, img) in enumerate(zip(tiles, images), 1):
            records.extend(segment_tiles([xy], [img], segment))
            records[-1]["tile"] = i - 1
            yield i
        return records

    def finish(records):
        targets = stage_targets(records, px, include_grid=True)
        widget._dish["targets"] = targets
        raw = sum(len(r["targets"]) for r in records)
        _show(widget, targets)
        _say(widget, f"{len(tiles)} tiles: {raw} detections -> {len(targets)} unique droplets "
                     f"({sum(len(t['grid']) for t in targets)} grid points). Review, then export.")
        for b in buttons:
            b.setEnabled(True)

    def failed(exc):
        _say(widget, f"segmentation failed -- {exc}")
        for b in buttons:
            b.setEnabled(True)

    worker = work()
    worker.yielded.connect(lambda i: _say(widget, f"pass 1: tile {i}/{len(tiles)}"))
    worker.returned.connect(finish)
    worker.errored.connect(failed)
    worker.start()
    widget._dish_worker = worker


def _show(widget, targets):
    """Review layers in stage um; napari points are (row, col) = (y, x)."""
    viewer = widget.viewer
    (cx, cy), r = well_geometry(widget)
    angles = np.linspace(0, 2 * np.pi, 120)
    ring = np.column_stack([cy + r * np.sin(angles), cx + r * np.cos(angles)])
    if WELL in viewer.layers:
        viewer.layers[WELL].data = [ring]
    else:
        viewer.add_shapes([ring], shape_type="polygon", name=WELL, edge_color="cyan",
                          face_color="transparent", edge_width=20)
    aim = np.array([[t["xy"][1], t["xy"][0]] for t in targets]).reshape(-1, 2)
    if REVIEW in viewer.layers:
        viewer.layers[REVIEW].data = aim
    else:
        viewer.add_points(aim, name=REVIEW, face_color="yellow", size=40, symbol="cross")


def _export(widget):
    targets = widget._dish.get("targets")
    if not targets:
        _say(widget, "nothing to export -- segment first")
        return
    path, _ = QFileDialog.getSaveFileName(widget, "Export dish targets", "dish_targets.csv", "CSV (*.csv)")
    if path:
        export_targets_csv(path, targets)
        _say(widget, f"wrote {len(targets)} targets to {path}")
