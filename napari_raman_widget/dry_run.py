"""Hardware-free droplet segmentation and Raman-grid dry run.

Adds a "Dry run" panel to the widget. It loads a saved image or z-stack into
napari, segments lipid droplets with an independently selected Cellpose model,
then shows raw and accepted masks, one aim point per droplet (the point farthest
from its edge) and a Raman grid inside each droplet.
No microscope, camera or DAQ is touched.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from qtpy.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel,
    QPushButton, QSpinBox, QVBoxLayout,
)

from .ui_helpers import make_collapsible

IMAGE, RAW_LABELS, LABELS, AIM, GRID, REJECTED, CHANGING, STABLE = (
    "BODIPY (dry run)", "segmentation raw (dry run)", "droplets accepted (dry run)",
    "droplet aim (dry run)", "droplet grid (dry run)", "droplet rejected (dry run)",
    "revisit changing (dry run)", "revisit stable controls (dry run)",
)


def _plane_to_targets(plane, masks, params):
    from raman_mda_engine.aiming.droplets import droplet_targets

    targets = droplet_targets(
        masks, density=params["density"], margin=params["margin"], image=plane,
        min_area=params["min_area"],
        min_rel_intensity=params["min_rel_intensity"],
        max_area=params["max_area"],
    )
    # Add simple QC fields used for conservative pilot ranking.
    image_arr = np.asarray(plane)
    for t in targets:
        region = masks == t["label"]
        t["area"] = int(region.sum())
        t["mean_intensity"] = float(image_arr[region].mean()) if region.any() else 0.0
    # A Raman aim point should have a real interior safety margin, not merely
    # fall somewhere inside a tiny/edge-hugging mask.
    return [t for t in targets if t["radius"] >= params["margin"]]


def _watershed_masks(plane, background_sigma=12.0, min_distance=3, threshold_scale=1.0):
    """Segment bright BODIPY puncta with background subtraction + watershed."""
    from scipy import ndimage as ndi
    from skimage import feature, filters, morphology, segmentation

    plane = np.asarray(plane, np.float32)
    lo, hi = np.percentile(plane, [1, 99.8])
    if hi <= lo:
        return np.zeros(plane.shape, np.uint32)
    norm = np.clip((plane - lo) / (hi - lo), 0, 1)
    background = ndi.gaussian_filter(norm, float(background_sigma))
    corrected = np.clip(norm - background, 0, None)
    smooth = ndi.gaussian_filter(corrected, 1.0)
    positive = smooth[smooth > 0]
    if positive.size == 0:
        return np.zeros(plane.shape, np.uint32)
    threshold = filters.threshold_otsu(positive) * float(threshold_scale)
    binary = smooth > threshold
    binary = morphology.remove_small_objects(binary, min_size=8)
    binary = morphology.remove_small_holes(binary, area_threshold=8)
    distance = ndi.distance_transform_edt(binary)
    coords = feature.peak_local_max(
        distance,
        min_distance=max(1, int(min_distance)),
        labels=binary,
        exclude_border=False,
    )
    markers = np.zeros(binary.shape, np.int32)
    if len(coords):
        markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
    else:
        markers, _ = ndi.label(binary)
    return segmentation.watershed(-distance, markers, mask=binary).astype(np.uint32)


def build_dry_run_controls(widget):
    """Create the panel, attach it to ``widget`` and return its group box."""
    box = make_collapsible("Dry run: droplet segmentation (no hardware)", expanded=False)
    lay = QVBoxLayout()
    lay.addWidget(QLabel(
        "Compare Cellpose with a BODIPY-specific threshold + watershed baseline. "
        "Choose custom BODIPY weights when available."))

    engine_row = QHBoxLayout()
    engine_row.addWidget(QLabel("Segmentation engine:"))
    widget.dry_engine_combo = QComboBox()
    widget.dry_engine_combo.addItem("BODIPY threshold + watershed", "watershed")
    widget.dry_engine_combo.addItem("Cellpose", "cellpose")
    engine_row.addWidget(widget.dry_engine_combo, 1)
    lay.addLayout(engine_row)

    from .widget import _cellpose_model_names

    model_row = QHBoxLayout()
    model_row.addWidget(QLabel("Droplet Cellpose model:"))
    widget.dry_cellpose_combo = QComboBox()
    for model_name in _cellpose_model_names():
        widget.dry_cellpose_combo.addItem(model_name, model_name)
    current_model = widget.sel_cellpose_combo.currentData() or widget.sel_cellpose_combo.currentText()
    current_index = widget.dry_cellpose_combo.findData(current_model)
    if current_index >= 0:
        widget.dry_cellpose_combo.setCurrentIndex(current_index)
    widget.dry_cellpose_browse = QPushButton("...")
    widget.dry_cellpose_browse.setFixedWidth(30)
    widget.dry_cellpose_browse.setToolTip("Select user-trained BODIPY Cellpose weights")
    widget.dry_cellpose_browse.clicked.connect(
        lambda _checked=False: widget.browse_cellpose_model(widget.dry_cellpose_combo)
    )
    model_row.addWidget(widget.dry_cellpose_combo, 1)
    model_row.addWidget(widget.dry_cellpose_browse)
    lay.addLayout(model_row)

    ws_bg_row = QHBoxLayout()
    ws_bg_row.addWidget(QLabel("Watershed background sigma (px):"))
    widget.dry_ws_bg_input = QDoubleSpinBox()
    widget.dry_ws_bg_input.setRange(1.0, 100.0)
    widget.dry_ws_bg_input.setValue(12.0)
    widget.dry_ws_bg_input.setSingleStep(1.0)
    ws_bg_row.addWidget(widget.dry_ws_bg_input)
    lay.addLayout(ws_bg_row)

    ws_md_row = QHBoxLayout()
    ws_md_row.addWidget(QLabel("Watershed min peak distance (px):"))
    widget.dry_ws_distance_input = QSpinBox()
    widget.dry_ws_distance_input.setRange(1, 100)
    widget.dry_ws_distance_input.setValue(4)
    ws_md_row.addWidget(widget.dry_ws_distance_input)
    lay.addLayout(ws_md_row)

    ws_thr_row = QHBoxLayout()
    ws_thr_row.addWidget(QLabel("Watershed threshold scale:"))
    widget.dry_ws_threshold_input = QDoubleSpinBox()
    widget.dry_ws_threshold_input.setRange(0.25, 4.0)
    widget.dry_ws_threshold_input.setValue(1.0)
    widget.dry_ws_threshold_input.setSingleStep(0.05)
    ws_thr_row.addWidget(widget.dry_ws_threshold_input)
    lay.addLayout(ws_thr_row)

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
    widget.dry_layer_btn = QPushButton("Use selected image layer")
    widget.dry_layer_btn.setToolTip(
        "Segment an image layer already in the viewer, such as a snapped camera\n"
        "frame. Aim points are only valid for the image they were computed on.")
    lay.addWidget(widget.dry_layer_btn)

    def row(label, w):
        r = QHBoxLayout(); r.addWidget(QLabel(label)); r.addWidget(w); lay.addLayout(r)

    widget.dry_rel_input = QDoubleSpinBox()
    widget.dry_rel_input.setRange(0.0, 1.0); widget.dry_rel_input.setSingleStep(0.05)
    widget.dry_rel_input.setValue(0.10)
    widget.dry_rel_input.setToolTip(
        "Targeting filter: drop masks whose mean BODIPY intensity is below this "
        "fraction of the brightest candidate mask")
    row("Min relative intensity:", widget.dry_rel_input)
    widget.dry_diameter_input = QDoubleSpinBox()
    widget.dry_diameter_input.setRange(0.0, 500.0)
    widget.dry_diameter_input.setValue(0.0)
    widget.dry_diameter_input.setSpecialValueText("Auto (model default)")
    widget.dry_diameter_input.setToolTip(
        "Droplet diameter in pixels. Leave on Auto: on BODIPY images a fixed\n"
        "diameter such as 15 found only 2 of 9 droplets.")
    row("Droplet diameter (px):", widget.dry_diameter_input)
    widget.dry_minarea_input = QSpinBox()
    widget.dry_minarea_input.setRange(1, 10_000_000)
    widget.dry_minarea_input.setSingleStep(10)
    widget.dry_minarea_input.setValue(50)
    widget.dry_minarea_input.setToolTip(
        "Targeting filter in pixels. Set this from the microscope pixel size "
        "and the smallest droplet you are willing to measure.")
    row("Min droplet area (px):", widget.dry_minarea_input)
    widget.dry_flow_input = QDoubleSpinBox()
    widget.dry_flow_input.setRange(0.0, 3.0)
    widget.dry_flow_input.setSingleStep(0.05)
    widget.dry_flow_input.setValue(0.4)
    widget.dry_flow_input.setToolTip(
        "Cellpose flow threshold. Lower is stricter; 0.4 is the Cellpose default. "
        "Increase only when reviewed droplets are being rejected.")
    row("Flow threshold:", widget.dry_flow_input)
    widget.dry_cellprob_input = QDoubleSpinBox()
    widget.dry_cellprob_input.setRange(-10.0, 10.0)
    widget.dry_cellprob_input.setSingleStep(0.25)
    widget.dry_cellprob_input.setValue(0.0)
    widget.dry_cellprob_input.setToolTip(
        "Cellpose cell-probability threshold. 0.0 is the model default; negative "
        "values accept weaker candidate regions and can create hazy/merged masks.")
    row("Cell probability threshold:", widget.dry_cellprob_input)
    widget.dry_maxarea_input = QSpinBox()
    widget.dry_maxarea_input.setRange(0, 10_000_000)
    widget.dry_maxarea_input.setSingleStep(500)
    widget.dry_maxarea_input.setValue(3000)
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
    widget.dry_margin_input.setRange(0.0, 100.0); widget.dry_margin_input.setValue(4.0)
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
    lay.addWidget(QLabel(
        "Send to acquisition: run 'Manual selection' first, then pick the\n"
        "cell layer. Targets must come from the camera's own image."))
    widget.dry_grid_check = QCheckBox("Include grid points (not just aim points)")
    widget.dry_grid_check.setChecked(False)
    lay.addWidget(widget.dry_grid_check)

    pilot_row = QHBoxLayout()
    widget.dry_limit_check = QCheckBox("Limit pilot to safest targets")
    widget.dry_limit_check.setChecked(True)
    widget.dry_limit_input = QSpinBox()
    widget.dry_limit_input.setRange(1, 1000)
    widget.dry_limit_input.setValue(10)
    widget.dry_limit_input.setSuffix(" targets")
    pilot_row.addWidget(widget.dry_limit_check)
    pilot_row.addWidget(widget.dry_limit_input)
    lay.addLayout(pilot_row)

    review_row = QHBoxLayout()
    widget.dry_reject_btn = QPushButton("Reject selected aim(s)")
    widget.dry_keep_btn = QPushButton("Keep selected aim(s) only")
    review_row.addWidget(widget.dry_reject_btn)
    review_row.addWidget(widget.dry_keep_btn)
    lay.addLayout(review_row)
    widget.dry_reset_btn = QPushButton("Reset acceptance")
    lay.addWidget(widget.dry_reset_btn)

    lay.addWidget(QLabel("Adaptive revisit selection:"))
    widget.dry_baseline_btn = QPushButton("Set accepted targets as baseline")
    lay.addWidget(widget.dry_baseline_btn)

    adaptive_match_row = QHBoxLayout()
    adaptive_match_row.addWidget(QLabel("Max match distance (px):"))
    widget.dry_match_distance_input = QDoubleSpinBox()
    widget.dry_match_distance_input.setRange(1.0, 500.0)
    widget.dry_match_distance_input.setValue(25.0)
    widget.dry_match_distance_input.setSingleStep(1.0)
    adaptive_match_row.addWidget(widget.dry_match_distance_input)
    lay.addLayout(adaptive_match_row)

    adaptive_change_row = QHBoxLayout()
    adaptive_change_row.addWidget(QLabel("Changing if |area change| >= (%):"))
    widget.dry_change_threshold_input = QDoubleSpinBox()
    widget.dry_change_threshold_input.setRange(0.0, 500.0)
    widget.dry_change_threshold_input.setValue(15.0)
    widget.dry_change_threshold_input.setSingleStep(1.0)
    adaptive_change_row.addWidget(widget.dry_change_threshold_input)
    lay.addLayout(adaptive_change_row)

    adaptive_stable_row = QHBoxLayout()
    adaptive_stable_row.addWidget(QLabel("Stable control if |area change| <= (%):"))
    widget.dry_stable_threshold_input = QDoubleSpinBox()
    widget.dry_stable_threshold_input.setRange(0.0, 100.0)
    widget.dry_stable_threshold_input.setValue(5.0)
    widget.dry_stable_threshold_input.setSingleStep(1.0)
    adaptive_stable_row.addWidget(widget.dry_stable_threshold_input)
    lay.addLayout(adaptive_stable_row)

    adaptive_count_row = QHBoxLayout()
    adaptive_count_row.addWidget(QLabel("Changing targets:"))
    widget.dry_change_count_input = QSpinBox()
    widget.dry_change_count_input.setRange(1, 1000)
    widget.dry_change_count_input.setValue(5)
    adaptive_count_row.addWidget(widget.dry_change_count_input)
    adaptive_count_row.addWidget(QLabel("Stable controls:"))
    widget.dry_stable_count_input = QSpinBox()
    widget.dry_stable_count_input.setRange(0, 1000)
    widget.dry_stable_count_input.setValue(2)
    adaptive_count_row.addWidget(widget.dry_stable_count_input)
    lay.addLayout(adaptive_count_row)

    widget.dry_auto_adaptive_check = QCheckBox(
        "Auto-prioritize after each new segmentation when baseline exists"
    )
    widget.dry_auto_adaptive_check.setChecked(True)
    lay.addWidget(widget.dry_auto_adaptive_check)
    widget.dry_prioritize_btn = QPushButton(
        "Prioritize changing droplets + stable controls"
    )
    lay.addWidget(widget.dry_prioritize_btn)
    widget.dry_adaptive_status = QLabel("No adaptive baseline set")
    widget.dry_adaptive_status.setWordWrap(True)
    lay.addWidget(widget.dry_adaptive_status)

    widget.dry_best_check = QCheckBox("Best plane per droplet only")
    widget.dry_best_check.setChecked(True)
    lay.addWidget(widget.dry_best_check)
    pos_row = QHBoxLayout()
    pos_row.addWidget(QLabel("Stage position index:"))
    widget.dry_pos_input = QSpinBox()
    widget.dry_pos_input.setRange(0, 100000)
    pos_row.addWidget(widget.dry_pos_input)
    lay.addLayout(pos_row)
    widget.dry_validate_map_btn = QPushButton("Validate image -> galvo mapping (no motion)")
    lay.addWidget(widget.dry_validate_map_btn)
    widget.dry_prepare_mda_btn = QPushButton("Prepare curated droplets for Raman MDA")
    lay.addWidget(widget.dry_prepare_mda_btn)
    widget.dry_send_btn = QPushButton("Legacy: send to selected cell layer")
    widget.dry_send_btn.hide()
    lay.addWidget(widget.dry_send_btn)
    widget.dry_status = QLabel("")
    widget.dry_status.setWordWrap(True)
    lay.addWidget(widget.dry_status)
    box.setLayout(lay)

    widget._dry = {
        "stack": None,
        "raw": None,
        "targets": {},
        "all_targets": {},
        "raw_masks": {},
        "rejected": set(),
        "aim_index_keys": [],
        "baseline": [],
        "adaptive_matches": [],
        "adaptive_keep": None,
    }
    widget.dry_open_btn.clicked.connect(lambda _=False: _open(widget))
    widget.dry_plane_btn.clicked.connect(lambda _=False: _segment(widget, current_only=True))
    widget.dry_all_btn.clicked.connect(lambda _=False: _segment(widget, current_only=False))
    widget.dry_export_btn.clicked.connect(lambda _=False: _export(widget))
    widget.dry_layer_btn.clicked.connect(lambda _=False: _use_layer(widget))
    widget.dry_send_btn.clicked.connect(lambda _=False: _send(widget))
    widget.dry_validate_map_btn.clicked.connect(lambda _=False: _validate_galvo_mapping(widget))
    widget.dry_prepare_mda_btn.clicked.connect(lambda _=False: _prepare_raman_mda(widget))
    widget.dry_reject_btn.clicked.connect(lambda _=False: _reject_selected(widget))
    widget.dry_keep_btn.clicked.connect(lambda _=False: _keep_selected_only(widget))
    widget.dry_reset_btn.clicked.connect(lambda _=False: _reset_acceptance(widget))
    widget.dry_baseline_btn.clicked.connect(lambda _=False: _set_adaptive_baseline(widget))
    widget.dry_prioritize_btn.clicked.connect(lambda _=False: _adaptive_prioritize(widget))
    widget.dry_grid_check.toggled.connect(lambda _=False: _refresh_target_layers(widget))
    widget.dry_limit_check.toggled.connect(lambda _=False: _apply_pilot_limit(widget))
    widget.dry_limit_input.valueChanged.connect(lambda _=0: _apply_pilot_limit(widget))
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
    _load_stack(widget, stack, Path(path).name)


def _load_stack(widget, stack, label):
    """Show ``stack`` (Z, Y, X) in the viewer with empty result layers."""
    stack = np.asarray(stack).astype(np.float32)
    for name in (
        IMAGE, RAW_LABELS, LABELS, AIM, GRID, REJECTED, CHANGING, STABLE
    ):
        if name in widget.viewer.layers:
            widget.viewer.layers.remove(name)
    widget._dry.update(
        stack=stack, targets={}, all_targets={}, raw_masks={},
        rejected=set(), aim_index_keys=[], adaptive_matches=[],
        adaptive_keep=None,
    )
    widget.viewer.add_image(
        stack, name=IMAGE, colormap="green", blending="additive",
        contrast_limits=[float(v) for v in np.percentile(stack, [1, 99.8])])
    widget.viewer.add_labels(np.zeros(stack.shape, np.uint32), name=RAW_LABELS, opacity=0.25)
    widget.viewer.add_labels(np.zeros(stack.shape, np.uint32), name=LABELS, opacity=0.55)
    widget.viewer.add_points(
        np.empty((0, 3)), name=AIM, size=8, face_color="red", ndim=3
    )
    widget.viewer.add_points(
        np.empty((0, 3)), name=GRID, size=2, face_color="yellow", ndim=3
    )
    widget.viewer.add_points(
        np.empty((0, 3)), name=REJECTED, size=7, face_color="gray", ndim=3
    )
    widget.viewer.add_points(
        np.empty((0, 3)), name=CHANGING, size=12, face_color="magenta", ndim=3
    )
    widget.viewer.add_points(
        np.empty((0, 3)), name=STABLE, size=10, face_color="cyan", ndim=3
    )
    baseline_note = (
        f"; adaptive baseline has {len(widget._dry.get('baseline', []))} droplet(s)"
        if widget._dry.get("baseline") else ""
    )
    _say(widget, f"loaded {label}: {stack.shape[0]} plane(s), "
                 f"{stack.shape[1]}x{stack.shape[2]} px{baseline_note}")


def _use_layer(widget):
    """Segment an image already in the viewer (e.g. a snapped camera frame)."""
    from napari.layers import Image

    layer = widget.viewer.layers.selection.active
    if not isinstance(layer, Image) or layer.name == IMAGE or layer.rgb:
        _say(widget, "select a grayscale image layer in the layer list first "
                     "(not the dry-run copy)")
        return
    data = np.asarray(layer.data)
    if data.ndim == 2:
        data = data[None]
    elif data.ndim != 3:
        _say(widget, f"layer '{layer.name}' has shape {data.shape}; need 2D or Z-Y-X")
        return
    _load_stack(widget, data, f"layer '{layer.name}'")


def _segment(widget, current_only):
    from napari.qt.threading import thread_worker

    stack = widget._dry["stack"]
    if stack is None:
        _say(widget, "open an image or z-stack first")
        return
    engine = widget.dry_engine_combo.currentData() or "watershed"
    model_name = None
    if engine == "cellpose":
        try:
            model_name = widget._selected_cellpose_model(widget.dry_cellpose_combo)
        except (FileNotFoundError, ValueError) as exc:
            _say(widget, str(exc))
            return
    from .widget import _cellpose_diameter

    diameter = _cellpose_diameter(widget.dry_diameter_input.value())
    params = dict(
        density=int(widget.dry_density_input.value()),
        margin=float(widget.dry_margin_input.value()),
        flow=float(widget.dry_flow_input.value()),
        cellprob=float(widget.dry_cellprob_input.value()),
        min_area=int(widget.dry_minarea_input.value()),
        min_rel_intensity=float(widget.dry_rel_input.value()),
        max_area=int(widget.dry_maxarea_input.value()) or None,
        ws_background_sigma=float(widget.dry_ws_bg_input.value()),
        ws_min_distance=int(widget.dry_ws_distance_input.value()),
        ws_threshold_scale=float(widget.dry_ws_threshold_input.value()),
    )
    zs = ([int(widget.viewer.dims.current_step[0])] if current_only and stack.shape[0] > 1
          else list(range(stack.shape[0])))
    zs = [min(max(z, 0), stack.shape[0] - 1) for z in zs]
    link = widget.dry_link_check.isChecked() and len(zs) > 1
    raw_labels_layer = widget.viewer.layers[RAW_LABELS]
    labels_layer = widget.viewer.layers[LABELS]
    raw_labels_layer.data = np.zeros(stack.shape, np.uint32)
    labels_layer.data = np.zeros(stack.shape, np.uint32)
    for name in (AIM, GRID, REJECTED, CHANGING, STABLE):
        widget.viewer.layers[name].data = np.empty((0, 3))
    widget._dry["targets"] = {}
    widget._dry["all_targets"] = {}
    widget._dry["raw_masks"] = {}
    widget._dry["rejected"] = set()
    widget._dry["aim_index_keys"] = []
    for b in (widget.dry_plane_btn, widget.dry_all_btn, widget.dry_open_btn, widget.dry_layer_btn):
        b.setEnabled(False)

    @thread_worker
    def work():
        model = None
        eval_cellpose_model = None
        if engine == "cellpose":
            from raman_mda_engine.aiming._cellpose import (
                eval_cellpose_model as _eval_cellpose_model,
                new_cellpose_model,
            )

            yield ("msg", f"loading Cellpose model: {model_name}...")
            model = new_cellpose_model(model_name)
            eval_cellpose_model = _eval_cellpose_model
        else:
            yield ("msg", "running BODIPY threshold + watershed...")

        for n, z in enumerate(zs, 1):
            plane = stack[z]
            lo, hi = np.percentile(plane, [1, 99.8])
            if hi <= lo:
                empty = np.zeros(plane.shape, np.uint32)
                yield ("plane", z, empty, empty, [], n)
                continue

            if engine == "cellpose":
                norm = np.clip((plane - lo) / (hi - lo), 0, 1)
                masks = eval_cellpose_model(
                    model,
                    norm,
                    diameter=diameter,
                    flow_threshold=params["flow"],
                    cellprob_threshold=params["cellprob"],
                    normalize=False,
                )
                masks = masks[0] if isinstance(masks, tuple) else masks
            else:
                masks = _watershed_masks(
                    plane,
                    background_sigma=params["ws_background_sigma"],
                    min_distance=params["ws_min_distance"],
                    threshold_scale=params["ws_threshold_scale"],
                )

            targets = _plane_to_targets(plane, masks, params)
            keep = np.isin(masks, [t["label"] for t in targets])
            accepted = np.where(keep, masks, 0).astype(np.uint32)
            yield ("plane", z, masks.astype(np.uint32), accepted, targets, n)

    def finish(_=None):
        if link:
            from raman_mda_engine.aiming.droplets import link_labels_across_z

            raw = np.stack([widget._dry["raw_masks"][z] for z in zs])
            linked = link_labels_across_z(raw)
            labels_layer.data = np.zeros(stack.shape, np.uint32)
            for i, z in enumerate(zs):
                labels_layer.data[z] = linked[i]
                # carry the linked id onto each candidate target
                for t in widget._dry["all_targets"][z]:
                    t["track"] = int(linked[i][tuple(t["aim"])])
            labels_layer.refresh()
        if (
            widget._dry.get("baseline")
            and widget.dry_auto_adaptive_check.isChecked()
        ):
            _adaptive_prioritize(widget)
        else:
            _apply_pilot_limit(widget)
        total = sum(len(v) for v in widget._dry["all_targets"].values())
        accepted = sum(len(v) for v in widget._dry["targets"].values())
        _say(
            widget,
            f"done: {accepted} accepted target(s) from {total} candidate droplet(s) "
            f"over {len(zs)} plane(s)",
        )
        for b in (widget.dry_plane_btn, widget.dry_all_btn, widget.dry_open_btn, widget.dry_layer_btn):
            b.setEnabled(True)

    def failed(exc):
        _say(widget, f"segmentation failed -- {exc}")
        for b in (widget.dry_plane_btn, widget.dry_all_btn, widget.dry_open_btn, widget.dry_layer_btn):
            b.setEnabled(True)

    def on_yield(msg):
        if msg[0] == "msg":
            _say(widget, msg[1])
            return
        _, z, raw_masks, masks, targets, n = msg
        # Linking across Z uses the accepted droplet masks, while the separate
        # RAW_LABELS layer remains available for diagnosing Cellpose itself.
        widget._dry["raw_masks"][z] = masks
        widget._dry["all_targets"][z] = targets
        raw_labels_layer.data[z] = raw_masks
        labels_layer.data[z] = masks
        raw_labels_layer.refresh()
        labels_layer.refresh()
        if stack.shape[0] > 1:
            widget.viewer.dims.set_point(0, z)
        _say(widget, f"plane {z}: {len(targets)} candidate droplet(s)  [{n}/{len(zs)}]")

    worker = work()
    worker.yielded.connect(on_yield)
    worker.returned.connect(finish)
    worker.errored.connect(failed)
    worker.start()
    widget._dry_worker = worker


def _target_key(z, target):
    """Stable identifier for one candidate target in the dry-run session."""
    return (int(z), int(target.get("track", target["label"])))


def _candidate_entries(widget):
    """Return (key, z, target) for every post-segmentation candidate."""
    entries = []
    for z in sorted(widget._dry.get("all_targets", {})):
        for target in widget._dry["all_targets"][z]:
            entries.append((_target_key(z, target), z, target))
    return entries


def _refresh_target_layers(widget):
    """Rebuild accepted/rejected masks and points from the curation state."""
    stack = widget._dry.get("stack")
    if stack is None or LABELS not in widget.viewer.layers:
        return

    entries = _candidate_entries(widget)
    rejected_manual = set(widget._dry.get("rejected", set()))
    available = [item for item in entries if item[0] not in rejected_manual]

    # Conservative pilot ranking: prioritize the point with the largest safe
    # interior radius, then stronger BODIPY signal, then larger segmented area.
    ranked = sorted(
        available,
        key=lambda item: (
            float(item[2].get("radius", 0.0)),
            float(item[2].get("mean_intensity", 0.0)),
            float(item[2].get("area", 0.0)),
        ),
        reverse=True,
    )
    adaptive_keep = widget._dry.get("adaptive_keep")
    if adaptive_keep is not None:
        keep_keys = {item[0] for item in available if item[0] in adaptive_keep}
    elif widget.dry_limit_check.isChecked():
        keep_keys = {item[0] for item in ranked[: int(widget.dry_limit_input.value())]}
    else:
        keep_keys = {item[0] for item in ranked}

    accepted = {}
    excluded = []
    aim_points = []
    aim_keys = []
    grid_points = []
    accepted_masks = np.zeros(stack.shape, np.uint32)

    for key, z, target in entries:
        if key in keep_keys:
            accepted.setdefault(z, []).append(target)
            aim_points.append([z, *target["aim"]])
            aim_keys.append(key)
            if widget.dry_grid_check.isChecked() and len(target.get("grid", [])):
                grid_points.extend([[z, *p] for p in target["grid"]])

            base_masks = widget._dry.get("raw_masks", {}).get(z)
            if base_masks is not None:
                accepted_masks[z][base_masks == target["label"]] = target["label"]
        else:
            excluded.append([z, *target["aim"]])

    widget._dry["targets"] = accepted
    widget._dry["aim_index_keys"] = aim_keys

    widget.viewer.layers[LABELS].data = accepted_masks
    widget.viewer.layers[LABELS].refresh()
    widget.viewer.layers[AIM].data = (
        np.asarray(aim_points, float).reshape(-1, 3)
        if aim_points else np.empty((0, 3))
    )
    widget.viewer.layers[GRID].data = (
        np.asarray(grid_points, float).reshape(-1, 3)
        if grid_points else np.empty((0, 3))
    )
    widget.viewer.layers[REJECTED].data = (
        np.asarray(excluded, float).reshape(-1, 3)
        if excluded else np.empty((0, 3))
    )

    total = len(entries)
    kept = len(aim_points)
    held = total - kept
    _say(widget, f"curated targets: {kept} accepted / {total} candidates; {held} held back")


def _apply_pilot_limit(widget):
    """Apply current safety ranking/limit and refresh all target layers."""
    _refresh_target_layers(widget)


def _selected_aim_keys(widget):
    if AIM not in widget.viewer.layers:
        return set()
    layer = widget.viewer.layers[AIM]
    selected = sorted(getattr(layer, "selected_data", set()))
    keys = widget._dry.get("aim_index_keys", [])
    return {keys[i] for i in selected if 0 <= i < len(keys)}


def _reject_selected(widget):
    selected = _selected_aim_keys(widget)
    if not selected:
        _say(widget, "select one or more red aim points first")
        return
    widget._dry.setdefault("rejected", set()).update(selected)
    _refresh_target_layers(widget)


def _keep_selected_only(widget):
    selected = _selected_aim_keys(widget)
    if not selected:
        _say(widget, "select one or more red aim points first")
        return
    all_keys = {key for key, _z, _t in _candidate_entries(widget)}
    widget._dry["rejected"] = all_keys - selected
    _refresh_target_layers(widget)


def _reset_acceptance(widget):
    widget._dry["rejected"] = set()
    widget._dry["adaptive_keep"] = None
    widget._dry["adaptive_matches"] = []
    if CHANGING in widget.viewer.layers:
        widget.viewer.layers[CHANGING].data = np.empty((0, 3))
    if STABLE in widget.viewer.layers:
        widget.viewer.layers[STABLE].data = np.empty((0, 3))
    _refresh_target_layers(widget)


def _set_adaptive_baseline(widget):
    """Store the currently accepted droplets as the morphology baseline."""
    targets = widget._dry.get("targets") or {}
    baseline = []
    for z in sorted(targets):
        for target in targets[z]:
            baseline.append(
                {
                    "z": int(z),
                    "aim": np.asarray(target["aim"], dtype=float).copy(),
                    "area": float(target.get("area", 0.0)),
                    "radius": float(target.get("radius", 0.0)),
                    "mean_intensity": float(target.get("mean_intensity", 0.0)),
                }
            )
    if not baseline:
        _say(widget, "no accepted droplets to store as baseline")
        return
    widget._dry["baseline"] = baseline
    widget._dry["adaptive_matches"] = []
    widget._dry["adaptive_keep"] = None
    widget.dry_adaptive_status.setText(
        f"Baseline stored: {len(baseline)} accepted droplet(s). "
        "Load/segment the next frame to prioritize revisits."
    )
    _say(widget, f"adaptive baseline stored: {len(baseline)} droplet(s)")


def _match_adaptive_targets(widget):
    """One-to-one nearest-neighbor match baseline to current candidates."""
    baseline = list(widget._dry.get("baseline") or [])
    current = _candidate_entries(widget)
    if not baseline or not current:
        return []

    max_distance = float(widget.dry_match_distance_input.value())
    candidates = []
    for bi, base in enumerate(baseline):
        for key, z, target in current:
            if int(z) != int(base["z"]):
                continue
            distance = float(
                np.linalg.norm(
                    np.asarray(target["aim"], dtype=float)
                    - np.asarray(base["aim"], dtype=float)
                )
            )
            if distance <= max_distance:
                candidates.append((distance, bi, key, z, target))

    matches = []
    used_baseline = set()
    used_current = set()
    for distance, bi, key, z, target in sorted(candidates, key=lambda row: row[0]):
        if bi in used_baseline or key in used_current:
            continue
        used_baseline.add(bi)
        used_current.add(key)
        base = baseline[bi]
        base_area = float(base.get("area", 0.0))
        current_area = float(target.get("area", 0.0))
        area_change_pct = (
            100.0 * (current_area - base_area) / base_area
            if base_area > 0
            else np.nan
        )
        matches.append(
            {
                "key": key,
                "z": int(z),
                "target": target,
                "baseline_index": int(bi),
                "distance_px": distance,
                "baseline_area": base_area,
                "current_area": current_area,
                "area_change_pct": float(area_change_pct),
                "abs_area_change_pct": float(abs(area_change_pct)),
            }
        )
    return matches


def _adaptive_prioritize(widget):
    """Select changing droplets plus stable matched controls for revisit."""
    if not widget._dry.get("baseline"):
        widget.dry_adaptive_status.setText("No adaptive baseline set")
        _say(widget, "set accepted targets as baseline first")
        return

    matches = _match_adaptive_targets(widget)
    widget._dry["adaptive_matches"] = matches
    if not matches:
        widget._dry["adaptive_keep"] = set()
        _refresh_target_layers(widget)
        widget.dry_adaptive_status.setText(
            "No baseline droplets matched the current segmentation."
        )
        _say(widget, "adaptive revisit: no matched droplets")
        return

    change_threshold = float(widget.dry_change_threshold_input.value())
    stable_threshold = float(widget.dry_stable_threshold_input.value())
    n_change = int(widget.dry_change_count_input.value())
    n_stable = int(widget.dry_stable_count_input.value())

    changing = sorted(
        [
            match for match in matches
            if np.isfinite(match["abs_area_change_pct"])
            and match["abs_area_change_pct"] >= change_threshold
        ],
        key=lambda match: match["abs_area_change_pct"],
        reverse=True,
    )[:n_change]

    stable = sorted(
        [
            match for match in matches
            if np.isfinite(match["abs_area_change_pct"])
            and match["abs_area_change_pct"] <= stable_threshold
            and match["key"] not in {row["key"] for row in changing}
        ],
        key=lambda match: match["abs_area_change_pct"],
    )[:n_stable]

    keep = {row["key"] for row in changing}
    keep.update(row["key"] for row in stable)
    widget._dry["adaptive_keep"] = keep

    # Annotate targets so exported Raman coordinates retain morphology context.
    for row in matches:
        target = row["target"]
        target["baseline_area"] = row["baseline_area"]
        target["area_change_pct"] = row["area_change_pct"]
        target["match_distance_px"] = row["distance_px"]
        target["adaptive_class"] = (
            "changing" if row["key"] in {x["key"] for x in changing}
            else "stable_control" if row["key"] in {x["key"] for x in stable}
            else "matched_not_selected"
        )

    _refresh_target_layers(widget)

    changing_pts = np.asarray(
        [[row["z"], *row["target"]["aim"]] for row in changing], dtype=float
    ).reshape(-1, 3) if changing else np.empty((0, 3))
    stable_pts = np.asarray(
        [[row["z"], *row["target"]["aim"]] for row in stable], dtype=float
    ).reshape(-1, 3) if stable else np.empty((0, 3))
    if CHANGING in widget.viewer.layers:
        widget.viewer.layers[CHANGING].data = changing_pts
    if STABLE in widget.viewer.layers:
        widget.viewer.layers[STABLE].data = stable_pts

    widget.dry_adaptive_status.setText(
        f"Matched {len(matches)} baseline droplet(s): "
        f"{len(changing)} changing + {len(stable)} stable control(s) selected. "
        "Area change is a 2D morphology proxy, not lipid mass/volume."
    )
    _say(
        widget,
        f"adaptive revisit: {len(changing)} changing + "
        f"{len(stable)} stable control(s) from {len(matches)} matches",
    )


def _validate_galvo_mapping(widget):
    """Validate curated image points against the loaded galvo calibration.

    This computes voltages only. It does not write to the DAQ or move hardware.
    """
    targets = widget._dry.get("targets") or {}
    if not any(targets.values()):
        _say(widget, "nothing to validate -- segment/curate targets first")
        return False
    if getattr(widget, "transformer", None) is None:
        _say(widget, "no aiming transformer loaded -- connect/reload calibration first")
        return False

    mapped = []
    try:
        for z in sorted(targets):
            for target in targets[z]:
                y, x = target["aim"]
                volts = np.asarray(
                    widget._pt_to_volts(np.array([y, x], dtype=float)),
                    dtype=float,
                )
                if volts.shape != (1, 2) or not np.all(np.isfinite(volts)):
                    raise ValueError("transformer returned a non-finite or invalid voltage pair")
                vx, vy = map(float, volts[0])
                if abs(vx) > 1.6 or abs(vy) > 1.6:
                    raise ValueError(
                        f"target ({y}, {x}) maps outside +/-1.6 V: "
                        f"({vx:.4f}, {vy:.4f}) V"
                    )
                mapped.append((int(z), int(y), int(x), vx, vy))
    except Exception as exc:
        widget._dry["galvo_preview"] = []
        _say(widget, f"galvo mapping invalid -- {exc}")
        return False

    widget._dry["galvo_preview"] = mapped
    vx = [row[3] for row in mapped]
    vy = [row[4] for row in mapped]
    _say(
        widget,
        f"galvo mapping valid for {len(mapped)} target(s); "
        f"X {min(vx):.3f}..{max(vx):.3f} V, "
        f"Y {min(vy):.3f}..{max(vy):.3f} V; no hardware moved",
    )
    return True


def _prepare_raman_mda(widget):
    """Prepare curated droplets as a standard Raman-MDA target source.

    This creates/updates the selection source and attaches it to the MDA
    metadata. It does not start MDA, move the stage/galvos, or open shutters.
    """
    targets = widget._dry.get("targets") or {}
    if not any(targets.values()):
        _say(widget, "nothing to prepare -- segment/curate targets first")
        return
    if widget.core is None or widget.main_window is None:
        _say(widget, "connect the microscope first so the MDA context is available")
        return
    if not _validate_galvo_mapping(widget):
        return

    try:
        from cns_control.utils import create_point_sources, _get_seq_from_napari
        from .unified_mda import apply_prepared_positions

        source_name = "cell:curated droplets"
        if source_name in widget.viewer.layers:
            widget.viewer.layers.remove(source_name)

        # A zero-size 1x1 square leaves each anchor unchanged: exactly one
        # Raman coordinate per accepted droplet.
        point_transformer = widget._make_point_transformer(0.0, 1)
        sources = create_point_sources(
            widget.viewer,
            point_transformer,
            size=12,
            names=[source_name],
            colors=["#ff3333ff"],
            roles=["cell"],
            display_names=["curated droplets"],
            source_ids=["curated-droplets"],
        )
        source = sources[0]

        rows = []
        for z in sorted(targets):
            for target in targets[z]:
                y, x = target["aim"]
                rows.append([0, int(widget.dry_pos_input.value()), 0, 0, float(y), float(x)])
        for row in rows:
            source._points.add(row)

        new_seq = _get_seq_from_napari(widget.main_window, survey=False)
        widget.selection_results = {
            "selection_type": "droplet",
            "sources": sources,
            "autofocus_p": np.arange(len(new_seq.stage_positions)),
            "new_seq": new_seq,
            "batch": False,
            "pre_acq": False,
        }
        widget._reset_grid_regions(clear_grid_selection=False)
        widget._refresh_cell_layer_controls()
        widget._activate_selected_cell_layer()

        # Attach the prepared source/targets to the existing Raman MDA panel.
        # This remains a preparation step only; acquisition starts only from
        # the MDA Run control.
        apply_prepared_positions(widget)
        _say(
            widget,
            f"prepared {len(rows)} curated droplet target(s) for Raman MDA; "
            "review the MDA summary before Run",
        )
    except Exception as exc:
        _say(widget, f"could not prepare Raman MDA -- {exc}")


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
        w.writerow([
            "z", "droplet_id", "kind", "y_px", "x_px",
            "galvo_x_V", "galvo_y_V", "best_z",
            "area_px", "baseline_area_px", "area_change_pct",
            "match_distance_px", "adaptive_class",
        ])
        for z in sorted(targets):
            for t in targets[z]:
                droplet = t.get("track", t["label"])
                pts = [("aim", t["aim"])]
                if widget.dry_grid_check.isChecked():
                    pts += [("grid", p) for p in t["grid"]]
                for kind, (y, x) in pts:
                    vx = vy = ""
                    if have_volts:
                        try:
                            v = widget._pt_to_volts(np.array([y, x], float))[0]
                            vx, vy = f"{v[0]:.5f}", f"{v[1]:.5f}"
                        except Exception:
                            pass
                    w.writerow([
                        z,
                        droplet,
                        kind,
                        int(y),
                        int(x),
                        vx,
                        vy,
                        int(best[droplet][0] == z),
                        int(t.get("area", 0)),
                        t.get("baseline_area", ""),
                        t.get("area_change_pct", ""),
                        t.get("match_distance_px", ""),
                        t.get("adaptive_class", ""),
                    ])
                    rows += 1
    note = "" if have_volts else " (no transformer loaded, so voltages are blank)"
    _say(widget, f"exported {rows} points to {Path(path).name}{note}")


def _send(widget):
    """Add the droplet aim/grid points to the selected cell layer.

    Mirrors what the widget does after automated selection: one row
    ``[t, position, c, z, y, x]`` per point, broadcast over t, c and z. The
    cell layer comes from 'Manual selection', so everything after this
    (apply prepared positions, MDA, Raman acquisition) is the existing path.
    """
    from raman_mda_engine.aiming.droplets import droplet_point_rows

    targets = widget._dry.get("targets") or {}
    if not any(targets.values()):
        _say(widget, "nothing to send -- segment first")
        return
    if getattr(widget, "selection_results", None) is None:
        _say(widget, "run 'Manual selection' first to create the cell layer")
        return
    source = widget._selected_cell_source()
    if source is None:
        _say(widget, "select a cell layer in 'Cell selection layers'")
        return
    rows = droplet_point_rows(
        targets, include_grid=widget.dry_grid_check.isChecked(),
        best_plane_only=widget.dry_best_check.isChecked())
    pos = int(widget.dry_pos_input.value())
    pts = [[0, pos, 0, 0, y, x] for _z, _d, _kind, y, x in rows]
    transformer = getattr(source, "transformer", None)
    # The DAQ needs at least 2 samples per channel; the automated path adds a
    # repeated point when the aiming pattern does not multiply points.
    if len(pts) == 1 and getattr(transformer, "multiplier", 2) <= 1:
        pts.append(list(pts[0]))
    for row in pts:
        source._points.add(row)
    n_drop = len({d for _z, d, _k, _y, _x in rows})
    _say(widget, f"sent {len(pts)} point(s) for {n_drop} droplet(s) to position {pos}; "
                 "apply prepared positions to MDA when ready")
