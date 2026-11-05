"""Hardware-free droplet segmentation and Raman-grid dry run.

Adds a "Dry run" panel to the widget. It loads a saved image or z-stack into
napari, segments lipid droplets with the Cellpose model and diameter already
chosen in "Automated cell selection", then shows the labels, one aim point per
droplet (the point farthest from its edge) and a Raman grid inside each droplet.
No microscope, camera or DAQ is touched.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from qtpy.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QVBoxLayout,
)

from .ui_helpers import make_collapsible

IMAGE, LABELS, AIM, GRID = (
    "BODIPY (dry run)", "droplets (dry run)", "droplet aim (dry run)",
    "droplet grid (dry run)",
)


def _plane_to_targets(plane, masks, params):
    from raman_mda_engine.aiming.droplets import droplet_targets

    return droplet_targets(
        masks, density=params["density"], margin=params["margin"], image=plane,
        min_rel_intensity=params["min_rel_intensity"],
        max_area=params["max_area"],
    )


def build_dry_run_controls(widget):
    """Create the panel, attach it to ``widget`` and return its group box."""
    box = make_collapsible("Dry run: droplet segmentation (no hardware)", expanded=False)
    lay = QVBoxLayout()
    lay.addWidget(QLabel(
        "Uses the Cellpose model and diameter chosen in\n"
        "'Automated cell selection' (set diameter to 0 for the model default)."))

    open_row = QHBoxLayout()
    widget.dry_open_btn = QPushButton("Open image / z-stack...")
    widget.dry_channel_input = QSpinBox()
    widget.dry_channel_input.setRange(0, 15)
    widget.dry_channel_input.setValue(2)
    widget.dry_channel_input.setToolTip("Channel index for Z-C-Y-X files (BODIPY)")
    open_row.addWidget(widget.dry_open_btn, 1)
    open_row.addWidget(QLabel("Channel:"))
    open_row.addWidget(widget.dry_channel_input)
    lay.addLayout(open_row)

    def row(label, w):
        r = QHBoxLayout(); r.addWidget(QLabel(label)); r.addWidget(w); lay.addLayout(r)

    widget.dry_rel_input = QDoubleSpinBox()
    widget.dry_rel_input.setRange(0.0, 1.0); widget.dry_rel_input.setSingleStep(0.05)
    widget.dry_rel_input.setValue(0.0)
    widget.dry_rel_input.setToolTip(
        "Drop objects dimmer than this fraction of the brightest droplet")
    row("Min relative intensity:", widget.dry_rel_input)
    widget.dry_maxarea_input = QSpinBox()
    widget.dry_maxarea_input.setRange(0, 10_000_000)
    widget.dry_maxarea_input.setSingleStep(500)
    widget.dry_maxarea_input.setValue(8000)
    widget.dry_maxarea_input.setSpecialValueText("No limit")
    widget.dry_maxarea_input.setToolTip(
        "Skip objects larger than this many pixels (0 = no limit). Large hazy\n"
        "blobs are usually clusters of unresolved small droplets. Depends on\n"
        "magnification and image size; tune it on your own images.")
    row("Max droplet area (px):", widget.dry_maxarea_input)
    widget.dry_density_input = QSpinBox()
    widget.dry_density_input.setRange(1, 500); widget.dry_density_input.setValue(6)
    row("Grid spacing (px):", widget.dry_density_input)
    widget.dry_margin_input = QDoubleSpinBox()
    widget.dry_margin_input.setRange(0.0, 100.0); widget.dry_margin_input.setValue(2.0)
    row("Edge margin (px):", widget.dry_margin_input)
    widget.dry_link_check = QCheckBox("Link droplets across z (same colour)")
    widget.dry_link_check.setChecked(True)
    lay.addWidget(widget.dry_link_check)

    btn_row = QHBoxLayout()
    widget.dry_plane_btn = QPushButton("Segment current plane")
    widget.dry_all_btn = QPushButton("Segment all planes")
    btn_row.addWidget(widget.dry_plane_btn); btn_row.addWidget(widget.dry_all_btn)
    lay.addLayout(btn_row)
    widget.dry_export_btn = QPushButton("Export targets (CSV)")
    lay.addWidget(widget.dry_export_btn)
    widget.dry_status = QLabel("")
    widget.dry_status.setWordWrap(True)
    lay.addWidget(widget.dry_status)
    box.setLayout(lay)

    widget._dry = {"stack": None, "raw": None, "targets": {}}
    widget.dry_open_btn.clicked.connect(lambda _=False: _open(widget))
    widget.dry_plane_btn.clicked.connect(lambda _=False: _segment(widget, current_only=True))
    widget.dry_all_btn.clicked.connect(lambda _=False: _segment(widget, current_only=False))
    widget.dry_export_btn.clicked.connect(lambda _=False: _export(widget))
    return box


def _say(widget, text):
    widget.dry_status.setText(text)
    widget.status.setText(f"Status: {text}")


def _open(widget):
    import tifffile

    path, _ = QFileDialog.getOpenFileName(
        widget, "Open image or z-stack", str(Path.home() / "Downloads"),
        "TIFF images (*.tif *.tiff);;All files (*)")
    if not path:
        return
    try:
        stack = tifffile.imread(path)
        if stack.ndim == 4:                      # Z, C, Y, X
            channel = int(widget.dry_channel_input.value())
            if channel >= stack.shape[1]:
                raise ValueError(f"file has {stack.shape[1]} channels")
            stack = stack[:, channel]
        elif stack.ndim == 2:
            stack = stack[None]
        elif stack.ndim != 3:
            raise ValueError(f"unsupported shape {stack.shape}")
    except Exception as exc:
        _say(widget, f"could not open image -- {exc}")
        return
    stack = stack.astype(np.float32)
    for name in (IMAGE, LABELS, AIM, GRID):
        if name in widget.viewer.layers:
            widget.viewer.layers.remove(name)
    widget._dry.update(stack=stack, targets={})
    widget.viewer.add_image(
        stack, name=IMAGE, colormap="green", blending="additive",
        contrast_limits=[float(v) for v in np.percentile(stack, [1, 99.8])])
    widget.viewer.add_labels(np.zeros(stack.shape, np.uint32), name=LABELS)
    widget.viewer.add_points(np.empty((0, 3)), name=AIM, size=8, face_color="red", ndim=3)
    widget.viewer.add_points(np.empty((0, 3)), name=GRID, size=2, face_color="yellow", ndim=3)
    _say(widget, f"loaded {Path(path).name}: {stack.shape[0]} plane(s), "
                 f"{stack.shape[1]}x{stack.shape[2]} px")


def _segment(widget, current_only):
    from napari.qt.threading import thread_worker

    stack = widget._dry["stack"]
    if stack is None:
        _say(widget, "open an image or z-stack first")
        return
    try:
        model_name = widget._selected_cellpose_model(widget.sel_cellpose_combo)
    except (FileNotFoundError, ValueError) as exc:
        _say(widget, str(exc))
        return
    from .widget import _cellpose_diameter

    diameter = _cellpose_diameter(widget.sel_cell_diameter_input.value())
    params = dict(density=int(widget.dry_density_input.value()),
                  margin=float(widget.dry_margin_input.value()),
                  min_rel_intensity=float(widget.dry_rel_input.value()),
                  max_area=int(widget.dry_maxarea_input.value()) or None)
    zs = ([int(widget.viewer.dims.current_step[0])] if current_only and stack.shape[0] > 1
          else list(range(stack.shape[0])))
    zs = [min(max(z, 0), stack.shape[0] - 1) for z in zs]
    link = widget.dry_link_check.isChecked() and len(zs) > 1
    labels_layer = widget.viewer.layers[LABELS]
    labels_layer.data = np.zeros(stack.shape, np.uint32)
    for name in (AIM, GRID):
        widget.viewer.layers[name].data = np.empty((0, 3))
    widget._dry["targets"] = {}
    widget._dry["raw_masks"] = {}
    for b in (widget.dry_plane_btn, widget.dry_all_btn, widget.dry_open_btn):
        b.setEnabled(False)

    @thread_worker
    def work():
        from raman_mda_engine.aiming._cellpose import (
            eval_cellpose_model, new_cellpose_model)

        yield ("msg", "loading Cellpose model...")
        model = new_cellpose_model(model_name)
        for n, z in enumerate(zs, 1):
            plane = stack[z]
            lo, hi = np.percentile(plane, [1, 99.8])
            if hi <= lo:
                yield ("plane", z, np.zeros(plane.shape, np.uint32), [], n)
                continue
            norm = np.clip((plane - lo) / (hi - lo), 0, 1)
            masks = eval_cellpose_model(
                model, norm, diameter=diameter, flow_threshold=0.6,
                cellprob_threshold=-2, normalize=False)
            masks = masks[0] if isinstance(masks, tuple) else masks
            targets = _plane_to_targets(plane, masks, params)
            keep = np.isin(masks, [t["label"] for t in targets])
            yield ("plane", z, np.where(keep, masks, 0).astype(np.uint32), targets, n)

    def finish(_=None):
        if link:
            from raman_mda_engine.aiming.droplets import link_labels_across_z

            raw = np.stack([widget._dry["raw_masks"][z] for z in zs])
            linked = link_labels_across_z(raw)
            labels_layer.data = np.zeros(stack.shape, np.uint32)
            for i, z in enumerate(zs):
                labels_layer.data[z] = linked[i]
                # carry the linked id onto each target
                for t in widget._dry["targets"][z]:
                    t["track"] = int(linked[i][tuple(t["aim"])])
            labels_layer.refresh()
        total = sum(len(v) for v in widget._dry["targets"].values())
        _say(widget, f"done: {total} droplet(s) over {len(zs)} plane(s)")
        for b in (widget.dry_plane_btn, widget.dry_all_btn, widget.dry_open_btn):
            b.setEnabled(True)

    def failed(exc):
        _say(widget, f"segmentation failed -- {exc}")
        for b in (widget.dry_plane_btn, widget.dry_all_btn, widget.dry_open_btn):
            b.setEnabled(True)

    def on_yield(msg):
        if msg[0] == "msg":
            _say(widget, msg[1])
            return
        _, z, masks, targets, n = msg
        widget._dry["raw_masks"][z] = masks
        widget._dry["targets"][z] = targets
        labels_layer.data[z] = masks
        labels_layer.refresh()
        zc = lambda p: np.column_stack([np.full(len(p), z), p])  # noqa: E731
        if targets:
            aim = widget.viewer.layers[AIM]
            aim.data = np.vstack([aim.data, zc(np.array([t["aim"] for t in targets]))])
            pts = np.concatenate([t["grid"] for t in targets])
            if len(pts):
                grid = widget.viewer.layers[GRID]
                grid.data = np.vstack([grid.data, zc(pts)])
        if stack.shape[0] > 1:
            widget.viewer.dims.set_point(0, z)
        _say(widget, f"plane {z}: {len(targets)} droplet(s)  [{n}/{len(zs)}]")

    worker = work()
    worker.yielded.connect(on_yield)
    worker.returned.connect(finish)
    worker.errored.connect(failed)
    worker.start()
    widget._dry_worker = worker


def _export(widget):
    targets = widget._dry.get("targets") or {}
    if not any(targets.values()):
        _say(widget, "nothing to export -- segment first")
        return
    path, _ = QFileDialog.getSaveFileName(
        widget, "Export droplet targets", "droplet_targets.csv", "CSV (*.csv)")
    if not path:
        return
    have_volts = getattr(widget, "transformer", None) is not None
    # Best plane per droplet = the one where its inscribed radius is largest
    # (the widest cross-section, i.e. the droplet's equator).
    best = {}
    for z in targets:
        for t in targets[z]:
            d = t.get("track", t["label"])
            if d not in best or t["radius"] > best[d][1]:
                best[d] = (z, t["radius"])
    rows = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["z", "droplet_id", "kind", "y_px", "x_px", "galvo_x_V", "galvo_y_V", "best_z"])
        for z in sorted(targets):
            for t in targets[z]:
                droplet = t.get("track", t["label"])
                pts = [("aim", t["aim"])] + [("grid", p) for p in t["grid"]]
                for kind, (y, x) in pts:
                    vx = vy = ""
                    if have_volts:
                        try:
                            v = widget._pt_to_volts(np.array([y, x], float))[0]
                            vx, vy = f"{v[0]:.5f}", f"{v[1]:.5f}"
                        except Exception:
                            pass
                    w.writerow([z, droplet, kind, int(y), int(x), vx, vy, int(best[droplet][0] == z)])
                    rows += 1
    note = "" if have_volts else " (no transformer loaded, so voltages are blank)"
    _say(widget, f"exported {rows} points to {Path(path).name}{note}")
