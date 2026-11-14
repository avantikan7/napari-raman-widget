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
    """Create the four-step experimental live-lipolysis workflow."""
    from qtpy.QtWidgets import QGroupBox

    box = make_collapsible("Live Lipolysis Tracker (experimental)", expanded=True)
    lay = QVBoxLayout(box)

    warning = QLabel(
        "EXPERIMENTAL LIVE MODE — use after fixed-cell stage targeting has been validated."
    )
    warning.setWordWrap(True)
    warning.setStyleSheet(
        "QLabel { border: 1px solid #b78103; padding: 6px; font-weight: bold; }"
    )
    warning.setToolTip(
        "This preserves the existing live workflow. It does not alter Jeff's "
        "Autofocus, MDA, stage, galvo, shutter, filter or laser controls."
    )
    lay.addWidget(warning)

    intro = QLabel(
        "Baseline → Track → Analyze → Revisit. This panel selects biological "
        "targets; Jeff's shared MDA remains authoritative for acquisition and Run."
    )
    intro.setWordWrap(True)
    lay.addWidget(intro)

    mode_row = QHBoxLayout()
    mode_row.addWidget(QLabel("Interface:"))
    widget.dry_interface_mode = QComboBox()
    widget.dry_interface_mode.addItems(["Basic", "Expert"])
    widget.dry_interface_mode.setToolTip(
        "Basic hides low-level segmentation parameters. Expert reveals the "
        "watershed, Cellpose, pixel-area and interior-margin controls."
    )
    mode_row.addWidget(widget.dry_interface_mode)
    lay.addLayout(mode_row)

    widget.dry_state_label = QLabel()
    widget.dry_state_label.setWordWrap(True)
    widget.dry_state_label.setStyleSheet(
        "QLabel { border: 1px solid palette(mid); padding: 5px; "
        "font-family: monospace; }"
    )
    widget.dry_state_label.setToolTip(
        "Live state summary. New images, segmentation or curation changes "
        "invalidate previous mapping validation and MDA preparation."
    )
    lay.addWidget(widget.dry_state_label)

    def step(title, summary, tooltip):
        group = QGroupBox(title)
        glay = QVBoxLayout(group)
        note = QLabel(summary)
        note.setWordWrap(True)
        note.setToolTip(tooltip)
        glay.addWidget(note)
        lay.addWidget(group)
        return glay

    def subgroup(parent_layout, title):
        group = QGroupBox(title)
        glay = QVBoxLayout(group)
        parent_layout.addWidget(group)
        return glay

    def row(layout, label, control):
        r = QHBoxLayout()
        r.addWidget(QLabel(label))
        r.addWidget(control)
        layout.addLayout(r)

    # ---------------- 1. BASELINE ----------------
    baseline = step(
        "1 — Baseline",
        "Acquire or select T0, segment candidate LDs, review safe targets and "
        "store the accepted identities.",
        "T0 stores the reference field image plus accepted droplet measurements. "
        "Later frames are registered before one-to-one matching.",
    )

    seg = subgroup(baseline, "Acquire/select T0 and segment LDs")
    engine_row = QHBoxLayout()
    engine_row.addWidget(QLabel("Segmentation engine:"))
    widget.dry_engine_combo = QComboBox()
    widget.dry_engine_combo.addItem("BODIPY threshold + watershed", "watershed")
    widget.dry_engine_combo.addItem("Cellpose", "cellpose")
    engine_row.addWidget(widget.dry_engine_combo, 1)
    seg.addLayout(engine_row)

    from .widget import _cellpose_model_names

    model_row = QHBoxLayout()
    model_row.addWidget(QLabel("Droplet Cellpose model:"))
    widget.dry_cellpose_combo = QComboBox()
    for model_name in _cellpose_model_names():
        widget.dry_cellpose_combo.addItem(model_name, model_name)
    current_model = (
        widget.sel_cellpose_combo.currentData()
        or widget.sel_cellpose_combo.currentText()
    )
    current_index = widget.dry_cellpose_combo.findData(current_model)
    if current_index >= 0:
        widget.dry_cellpose_combo.setCurrentIndex(current_index)
    widget.dry_cellpose_browse = QPushButton("...")
    widget.dry_cellpose_browse.setFixedWidth(30)
    widget.dry_cellpose_browse.clicked.connect(
        lambda _checked=False: widget.browse_cellpose_model(widget.dry_cellpose_combo)
    )
    model_row.addWidget(widget.dry_cellpose_combo, 1)
    model_row.addWidget(widget.dry_cellpose_browse)
    seg.addLayout(model_row)

    open_row = QHBoxLayout()
    widget.dry_open_btn = QPushButton("Open image / z-stack...")
    widget.dry_channel_input = QSpinBox()
    widget.dry_channel_input.setRange(0, 15)
    widget.dry_channel_input.setValue(2)
    open_row.addWidget(widget.dry_open_btn, 1)
    open_row.addWidget(QLabel("Channel:"))
    open_row.addWidget(widget.dry_channel_input)
    seg.addLayout(open_row)

    widget.dry_layer_btn = QPushButton("Use selected image layer")
    seg.addWidget(widget.dry_layer_btn)

    widget.dry_advanced_box = make_collapsible(
        "Advanced segmentation parameters", expanded=False
    )
    advanced_layout = QVBoxLayout(widget.dry_advanced_box)

    widget.dry_ws_bg_input = QDoubleSpinBox()
    widget.dry_ws_bg_input.setRange(1.0, 100.0)
    widget.dry_ws_bg_input.setValue(12.0)
    widget.dry_ws_bg_input.setSingleStep(1.0)
    row(advanced_layout, "Watershed background sigma (px):", widget.dry_ws_bg_input)

    widget.dry_ws_distance_input = QSpinBox()
    widget.dry_ws_distance_input.setRange(1, 100)
    widget.dry_ws_distance_input.setValue(4)
    row(advanced_layout, "Watershed min peak distance (px):", widget.dry_ws_distance_input)

    widget.dry_ws_threshold_input = QDoubleSpinBox()
    widget.dry_ws_threshold_input.setRange(0.25, 4.0)
    widget.dry_ws_threshold_input.setValue(1.0)
    widget.dry_ws_threshold_input.setSingleStep(0.05)
    row(advanced_layout, "Watershed threshold scale:", widget.dry_ws_threshold_input)

    widget.dry_rel_input = QDoubleSpinBox()
    widget.dry_rel_input.setRange(0.0, 1.0)
    widget.dry_rel_input.setSingleStep(0.05)
    widget.dry_rel_input.setValue(0.10)
    row(advanced_layout, "Min relative intensity:", widget.dry_rel_input)

    widget.dry_diameter_input = QDoubleSpinBox()
    widget.dry_diameter_input.setRange(0.0, 500.0)
    widget.dry_diameter_input.setValue(0.0)
    widget.dry_diameter_input.setSpecialValueText("Auto (model default)")
    row(advanced_layout, "Droplet diameter (px):", widget.dry_diameter_input)

    widget.dry_minarea_input = QSpinBox()
    widget.dry_minarea_input.setRange(1, 10_000_000)
    widget.dry_minarea_input.setSingleStep(10)
    widget.dry_minarea_input.setValue(50)
    row(advanced_layout, "Min droplet area (px):", widget.dry_minarea_input)

    widget.dry_flow_input = QDoubleSpinBox()
    widget.dry_flow_input.setRange(0.0, 3.0)
    widget.dry_flow_input.setSingleStep(0.05)
    widget.dry_flow_input.setValue(0.4)
    row(advanced_layout, "Cellpose flow threshold:", widget.dry_flow_input)

    widget.dry_cellprob_input = QDoubleSpinBox()
    widget.dry_cellprob_input.setRange(-10.0, 10.0)
    widget.dry_cellprob_input.setSingleStep(0.25)
    widget.dry_cellprob_input.setValue(0.0)
    row(advanced_layout, "Cellpose probability threshold:", widget.dry_cellprob_input)

    widget.dry_maxarea_input = QSpinBox()
    widget.dry_maxarea_input.setRange(0, 10_000_000)
    widget.dry_maxarea_input.setSingleStep(500)
    widget.dry_maxarea_input.setValue(3000)
    widget.dry_maxarea_input.setSpecialValueText("No limit")
    row(advanced_layout, "Max droplet area (px):", widget.dry_maxarea_input)

    widget.dry_density_input = QSpinBox()
    widget.dry_density_input.setRange(1, 500)
    widget.dry_density_input.setValue(6)
    row(advanced_layout, "Grid spacing (px):", widget.dry_density_input)

    widget.dry_margin_input = QDoubleSpinBox()
    widget.dry_margin_input.setRange(0.0, 100.0)
    widget.dry_margin_input.setValue(4.0)
    row(advanced_layout, "Safe interior margin (px):", widget.dry_margin_input)

    widget.dry_link_check = QCheckBox("Link droplets across Z (same object)")
    widget.dry_link_check.setChecked(True)
    advanced_layout.addWidget(widget.dry_link_check)
    seg.addWidget(widget.dry_advanced_box)

    btn_row = QHBoxLayout()
    widget.dry_plane_btn = QPushButton("Segment current plane")
    widget.dry_all_btn = QPushButton("Segment all planes")
    btn_row.addWidget(widget.dry_plane_btn)
    btn_row.addWidget(widget.dry_all_btn)
    seg.addLayout(btn_row)

    qc = subgroup(baseline, "Review T0 targets and set identities")
    legend = QLabel(
        "Overlay legend: red = Raman aim | gray = held/rejected | "
        "magenta = changing revisit | cyan = stable-control revisit"
    )
    legend.setWordWrap(True)
    legend.setToolTip(
        "Stable means morphology-stable after tracking, not spatially stationary."
    )
    qc.addWidget(legend)

    widget.dry_grid_check = QCheckBox("Include grid points (not just one aim point)")
    widget.dry_grid_check.setChecked(False)
    qc.addWidget(widget.dry_grid_check)

    pilot_row = QHBoxLayout()
    widget.dry_limit_check = QCheckBox("Limit pilot to safest targets")
    widget.dry_limit_check.setChecked(True)
    widget.dry_limit_input = QSpinBox()
    widget.dry_limit_input.setRange(1, 1000)
    widget.dry_limit_input.setValue(10)
    widget.dry_limit_input.setSuffix(" targets")
    pilot_row.addWidget(widget.dry_limit_check)
    pilot_row.addWidget(widget.dry_limit_input)
    qc.addLayout(pilot_row)

    review_row = QHBoxLayout()
    widget.dry_reject_btn = QPushButton("Reject selected aim(s)")
    widget.dry_keep_btn = QPushButton("Keep selected aim(s) only")
    review_row.addWidget(widget.dry_reject_btn)
    review_row.addWidget(widget.dry_keep_btn)
    qc.addLayout(review_row)

    widget.dry_reset_btn = QPushButton("Reset acceptance")
    qc.addWidget(widget.dry_reset_btn)

    widget.dry_baseline_btn = QPushButton("Set accepted targets as T0 identities")
    widget.dry_baseline_btn.setMinimumHeight(34)
    qc.addWidget(widget.dry_baseline_btn)

    # ---------------- 2. TRACK ----------------
    track = step(
        "2 — Track",
        "Register the current field to T0 and match the same droplets one-to-one.",
        "Whole-field registration corrects XY drift first. Matching then uses "
        "registered distance plus weak area/intensity consistency terms."
    )

    widget.dry_register_btn = QPushButton("Register current frame to T0 baseline")
    widget.dry_register_btn.setEnabled(False)
    track.addWidget(widget.dry_register_btn)

    widget.dry_tracking_status = QLabel("No T0 baseline / field registration yet.")
    widget.dry_tracking_status.setWordWrap(True)
    track.addWidget(widget.dry_tracking_status)

    widget.dry_match_distance_input = QDoubleSpinBox()
    widget.dry_match_distance_input.setRange(1.0, 500.0)
    widget.dry_match_distance_input.setValue(25.0)
    widget.dry_match_distance_input.setSingleStep(1.0)
    row(track, "Max registered match distance (px):", widget.dry_match_distance_input)

    # ---------------- 3. ANALYZE ----------------
    analyze = step(
        "3 — Analyze response",
        "Quantify morphology change, rank responders and retain morphology-stable controls.",
        "Movement alone is not called lipolysis. Classification uses registered, "
        "identity-matched morphology change."
    )

    widget.dry_change_threshold_input = QDoubleSpinBox()
    widget.dry_change_threshold_input.setRange(0.0, 500.0)
    widget.dry_change_threshold_input.setValue(15.0)
    widget.dry_change_threshold_input.setSingleStep(1.0)
    row(analyze, "Changing if |area change| >= (%):", widget.dry_change_threshold_input)

    widget.dry_stable_threshold_input = QDoubleSpinBox()
    widget.dry_stable_threshold_input.setRange(0.0, 100.0)
    widget.dry_stable_threshold_input.setValue(5.0)
    widget.dry_stable_threshold_input.setSingleStep(1.0)
    row(analyze, "Stable control if |area change| <= (%):", widget.dry_stable_threshold_input)

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
    analyze.addLayout(adaptive_count_row)

    widget.dry_auto_adaptive_check = QCheckBox(
        "Auto-register + re-rank after each new segmentation when T0 exists"
    )
    widget.dry_auto_adaptive_check.setChecked(True)
    analyze.addWidget(widget.dry_auto_adaptive_check)

    widget.dry_prioritize_btn = QPushButton(
        "Match + rank responders and stable controls"
    )
    widget.dry_prioritize_btn.setMinimumHeight(34)
    analyze.addWidget(widget.dry_prioritize_btn)

    widget.dry_adaptive_status = QLabel("No live-tracking baseline set")
    widget.dry_adaptive_status.setWordWrap(True)
    analyze.addWidget(widget.dry_adaptive_status)

    math_note = QLabel(
        "Current response metric: %dA = 100 × (A_t − A_0) / A_0. "
        "Planned spherical volume proxy: V ≈ πd³/6; lipolytic flux ≈ dV/dt."
    )
    math_note.setWordWrap(True)
    math_note.setToolTip(
        "Area change is a morphology response score. Do not call it biochemical "
        "lipolysis until tracking and volume/flux measurements are validated."
    )
    analyze.addWidget(math_note)

    # ---------------- 4. REVISIT ----------------
    revisit = step(
        "4 — Revisit",
        "Choose responders and stable controls, then prepare their Raman targets "
        "for Jeff's shared MDA.",
        "The shared MDA remains responsible for Autofocus, acquisition settings, "
        "saving, Run and Cancel."
    )

    af_note = QLabel(
        "Autofocus reference: do not use an LD that may shrink or disappear. "
        "Use a stable low-structure region or Jeff's automatic reference selection."
    )
    af_note.setWordWrap(True)
    revisit.addWidget(af_note)

    widget.dry_best_check = QCheckBox("Best plane per droplet only")
    widget.dry_best_check.setChecked(True)
    revisit.addWidget(widget.dry_best_check)

    pos_row = QHBoxLayout()
    pos_row.addWidget(QLabel("Stage position index:"))
    widget.dry_pos_input = QSpinBox()
    widget.dry_pos_input.setRange(0, 100000)
    pos_row.addWidget(widget.dry_pos_input)
    revisit.addLayout(pos_row)

    widget.dry_validate_map_btn = QPushButton(
        "Validate image → galvo mapping (math only; no motion)"
    )
    revisit.addWidget(widget.dry_validate_map_btn)

    widget.dry_prepare_mda_btn = QPushButton(
        "Send revisit targets to shared Raman MDA"
    )
    widget.dry_prepare_mda_btn.setEnabled(False)
    revisit.addWidget(widget.dry_prepare_mda_btn)

    widget.dry_export_btn = QPushButton("Export target + tracking table (CSV)")
    revisit.addWidget(widget.dry_export_btn)

    loop_note = QLabel(
        "At the next timepoint: image → register → track → analyze → revisit."
    )
    loop_note.setWordWrap(True)
    revisit.addWidget(loop_note)

    widget.dry_send_btn = QPushButton("Legacy: send to selected cell layer")
    widget.dry_send_btn.hide()

    widget.dry_status = QLabel("")
    widget.dry_status.setWordWrap(True)
    lay.addWidget(widget.dry_status)

    widget._dry = {
        "stack": None,
        "raw": None,
        "targets": {},
        "all_targets": {},
        "raw_masks": {},
        "rejected": set(),
        "aim_index_keys": [],
        "baseline": [],
        "baseline_image": None,
        "field_shift_yx": None,
        "registration_error": None,
        "adaptive_matches": [],
        "adaptive_keep": None,
        "mapping_valid": False,
        "mda_prepared": False,
        "galvo_preview": [],
    }

    widget.dry_open_btn.clicked.connect(lambda _=False: _open(widget))
    widget.dry_plane_btn.clicked.connect(
        lambda _=False: _segment(widget, current_only=True)
    )
    widget.dry_all_btn.clicked.connect(
        lambda _=False: _segment(widget, current_only=False)
    )
    widget.dry_export_btn.clicked.connect(lambda _=False: _export(widget))
    widget.dry_layer_btn.clicked.connect(lambda _=False: _use_layer(widget))
    widget.dry_send_btn.clicked.connect(lambda _=False: _send(widget))
    widget.dry_validate_map_btn.clicked.connect(
        lambda _=False: _validate_galvo_mapping(widget)
    )
    widget.dry_prepare_mda_btn.clicked.connect(
        lambda _=False: _prepare_raman_mda(widget)
    )
    widget.dry_register_btn.clicked.connect(
        lambda _=False: _register_to_baseline(widget)
    )
    widget.dry_reject_btn.clicked.connect(lambda _=False: _reject_selected(widget))
    widget.dry_keep_btn.clicked.connect(
        lambda _=False: _keep_selected_only(widget)
    )
    widget.dry_reset_btn.clicked.connect(lambda _=False: _reset_acceptance(widget))
    widget.dry_baseline_btn.clicked.connect(
        lambda _=False: _set_adaptive_baseline(widget)
    )
    widget.dry_prioritize_btn.clicked.connect(
        lambda _=False: _adaptive_prioritize(widget)
    )
    widget.dry_grid_check.toggled.connect(
        lambda _=False: _refresh_target_layers(widget)
    )
    widget.dry_limit_check.toggled.connect(
        lambda _=False: _apply_pilot_limit(widget)
    )
    widget.dry_limit_input.valueChanged.connect(
        lambda _=0: _apply_pilot_limit(widget)
    )

    def _set_interface_mode(text):
        widget.dry_advanced_box.setVisible(text == "Expert")

    widget.dry_interface_mode.currentTextChanged.connect(_set_interface_mode)
    _set_interface_mode(widget.dry_interface_mode.currentText())
    _update_state_summary(widget)
    return box


def _update_state_summary(widget):
    """Refresh the compact live-lipolysis state display."""
    dry = getattr(widget, "_dry", {})
    candidates = sum(len(v) for v in (dry.get("all_targets") or {}).values())
    accepted = sum(len(v) for v in (dry.get("targets") or {}).values())
    baseline = len(dry.get("baseline") or [])
    shift = dry.get("field_shift_yx")
    shift_text = (
        "No"
        if shift is None
        else f"Yes (dy={float(shift[0]):+.1f}, dx={float(shift[1]):+.1f} px)"
    )
    mapping = "Yes" if dry.get("mapping_valid") else "No"
    prepared = "Yes" if dry.get("mda_prepared") else "No"
    if hasattr(widget, "dry_state_label"):
        widget.dry_state_label.setText(
            f"Candidates: {candidates} | Accepted: {accepted}\n"
            f"T0 baseline: {baseline if baseline else 'No'} | Registered: {shift_text}\n"
            f"Galvo mapping validated: {mapping} | MDA prepared: {prepared}"
        )
    if hasattr(widget, "dry_register_btn"):
        widget.dry_register_btn.setEnabled(bool(baseline and dry.get("stack") is not None))
    if hasattr(widget, "dry_prepare_mda_btn"):
        widget.dry_prepare_mda_btn.setEnabled(
            bool(dry.get("mapping_valid") and accepted)
        )
    try:
        from .workflow_docks import refresh_control_hub
        refresh_control_hub(widget)
    except Exception:
        pass


def _invalidate_handoff(widget):
    """Any target change invalidates old mapping/MDA preparation state."""
    widget._dry["mapping_valid"] = False
    widget._dry["mda_prepared"] = False
    widget._dry["galvo_preview"] = []
    if hasattr(widget, "dry_prepare_mda_btn"):
        widget.dry_prepare_mda_btn.setEnabled(False)
    _update_state_summary(widget)


def _projection(stack):
    """Return a robust 2D image for field registration."""
    data = np.asarray(stack, dtype=np.float32)
    if data.ndim == 2:
        return data
    if data.ndim != 3:
        raise ValueError(f"registration expects Y-X or Z-Y-X, got {data.shape}")
    return np.max(data, axis=0)


def _register_to_baseline(widget):
    """Estimate whole-field translation from current image to T0."""
    baseline = widget._dry.get("baseline_image")
    current = widget._dry.get("stack")
    if baseline is None or current is None:
        widget._dry["field_shift_yx"] = None
        if hasattr(widget, "dry_tracking_status"):
            widget.dry_tracking_status.setText(
                "Set a T0 baseline, then load the next frame before registration."
            )
        _update_state_summary(widget)
        return False
    current = _projection(current)
    baseline = np.asarray(baseline, dtype=np.float32)
    if current.shape != baseline.shape:
        widget._dry["field_shift_yx"] = None
        if hasattr(widget, "dry_tracking_status"):
            widget.dry_tracking_status.setText(
                f"Registration skipped: T0 shape {baseline.shape} != current {current.shape}."
            )
        _update_state_summary(widget)
        return False
    try:
        from skimage.registration import phase_cross_correlation
        shift, error, _phase = phase_cross_correlation(
            baseline, current, upsample_factor=10
        )
        shift = np.asarray(shift, dtype=float)
        if shift.shape != (2,) or not np.isfinite(shift).all():
            raise ValueError("registration returned an invalid XY shift")
        widget._dry["field_shift_yx"] = shift
        widget._dry["registration_error"] = float(error)
        if hasattr(widget, "dry_tracking_status"):
            widget.dry_tracking_status.setText(
                f"Field registered to T0: dy={shift[0]:+.2f} px, "
                f"dx={shift[1]:+.2f} px; phase-correlation error={float(error):.3g}. "
                "Droplet matching uses registered coordinates."
            )
        _update_state_summary(widget)
        return True
    except Exception as exc:
        widget._dry["field_shift_yx"] = None
        widget._dry["registration_error"] = None
        if hasattr(widget, "dry_tracking_status"):
            widget.dry_tracking_status.setText(f"Field registration failed: {exc}")
        _update_state_summary(widget)
        return False


def _say(widget, text):
    widget.dry_status.setText(text)
    widget.status.setText(f"Status: {text}")
    _update_state_summary(widget)
    _update_state_summary(widget)


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
        adaptive_keep=None, field_shift_yx=None, registration_error=None,
        mapping_valid=False, mda_prepared=False, galvo_preview=[],
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
    _invalidate_handoff(widget)
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
            _register_to_baseline(widget)
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
    _invalidate_handoff(widget)
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
    """Store accepted droplets and the current field image as the T0 baseline."""
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
        _say(widget, "no accepted droplets to store as T0 baseline")
        return
    stack = widget._dry.get("stack")
    if stack is None:
        _say(widget, "no image loaded for T0 baseline")
        return
    widget._dry["baseline"] = baseline
    widget._dry["baseline_image"] = _projection(stack).copy()
    widget._dry["adaptive_matches"] = []
    widget._dry["adaptive_keep"] = None
    widget._dry["field_shift_yx"] = np.zeros(2, dtype=float)
    widget._dry["registration_error"] = 0.0
    _invalidate_handoff(widget)
    widget.dry_adaptive_status.setText(
        f"T0 stored: {len(baseline)} accepted droplet(s). "
        "Load/segment T1, then register and match before ranking."
    )
    if hasattr(widget, "dry_tracking_status"):
        widget.dry_tracking_status.setText(
            "T0 field stored. Current shift is (0, 0) px by definition."
        )
    _say(widget, f"T0 baseline stored: {len(baseline)} droplet(s)")

def _match_adaptive_targets(widget):
    """Globally match T0 droplets to current candidates after field registration.

    The cost is dominated by registered XY distance, with weak area/intensity
    consistency terms. This permits biological motion while preventing one
    current droplet from being assigned to multiple T0 droplets.
    """
    baseline = list(widget._dry.get("baseline") or [])
    current = _candidate_entries(widget)
    if not baseline or not current:
        return []

    shift = np.asarray(
        widget._dry.get("field_shift_yx")
        if widget._dry.get("field_shift_yx") is not None
        else (0.0, 0.0),
        dtype=float,
    )
    max_distance = float(widget.dry_match_distance_input.value())
    from scipy.optimize import linear_sum_assignment

    cost = np.full((len(baseline), len(current)), 1e6, dtype=float)
    raw_distance = np.full_like(cost, np.nan)
    registered_distance = np.full_like(cost, np.nan)

    for bi, base in enumerate(baseline):
        base_aim = np.asarray(base["aim"], dtype=float)
        base_area = max(float(base.get("area", 0.0)), 1e-9)
        base_intensity = max(float(base.get("mean_intensity", 0.0)), 1e-9)
        for ci, (_key, z, target) in enumerate(current):
            if int(z) != int(base["z"]):
                continue
            aim = np.asarray(target["aim"], dtype=float)
            raw = float(np.linalg.norm(aim - base_aim))
            registered = float(np.linalg.norm((aim + shift) - base_aim))
            raw_distance[bi, ci] = raw
            registered_distance[bi, ci] = registered
            if registered > max_distance:
                continue
            area = max(float(target.get("area", 0.0)), 1e-9)
            intensity = max(float(target.get("mean_intensity", 0.0)), 1e-9)
            area_term = abs(np.log(area / base_area))
            intensity_term = abs(np.log(intensity / base_intensity))
            # Distance is the main identity cue. Morphology/intensity terms are
            # deliberately weak because lipolysis itself changes these quantities.
            cost[bi, ci] = (
                registered / max(max_distance, 1e-9)
                + 0.20 * area_term
                + 0.05 * intensity_term
            )

    rows, cols = linear_sum_assignment(cost)
    matches = []
    for bi, ci in zip(rows, cols):
        if cost[bi, ci] >= 1e5:
            continue
        key, z, target = current[ci]
        base = baseline[bi]
        base_area = float(base.get("area", 0.0))
        current_area = float(target.get("area", 0.0))
        area_change_pct = (
            100.0 * (current_area - base_area) / base_area
            if base_area > 0
            else np.nan
        )
        confidence = float(np.exp(-cost[bi, ci]))
        matches.append(
            {
                "key": key,
                "z": int(z),
                "target": target,
                "baseline_index": int(bi),
                "distance_px": float(registered_distance[bi, ci]),
                "raw_distance_px": float(raw_distance[bi, ci]),
                "registered_distance_px": float(registered_distance[bi, ci]),
                "tracking_confidence": confidence,
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

    if (
        widget._dry.get("baseline_image") is not None
        and widget._dry.get("stack") is not None
        and widget._dry.get("field_shift_yx") is None
    ):
        _register_to_baseline(widget)
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
        target["raw_displacement_px"] = row.get("raw_distance_px", row["distance_px"])
        target["registered_distance_px"] = row.get("registered_distance_px", row["distance_px"])
        target["tracking_confidence"] = row.get("tracking_confidence", np.nan)
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

    mean_conf = np.mean([row.get("tracking_confidence", np.nan) for row in matches])
    widget.dry_adaptive_status.setText(
        f"Matched {len(matches)} T0 droplet(s): "
        f"{len(changing)} changing + {len(stable)} morphology-stable control(s). "
        f"Mean tracking confidence={mean_conf:.2f}. "
        "Movement is tracked separately; area change is still only a 2D morphology proxy."
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
        widget._dry["mapping_valid"] = False
        widget._dry["mda_prepared"] = False
        _update_state_summary(widget)
        _say(widget, f"galvo mapping invalid -- {exc}")
        return False

    widget._dry["galvo_preview"] = mapped
    widget._dry["mapping_valid"] = True
    widget._dry["mda_prepared"] = False
    _update_state_summary(widget)
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
    if not widget._dry.get("mapping_valid"):
        _say(widget, "validate image -> galvo mapping before preparing Raman MDA")
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
        widget._dry["mda_prepared"] = True
        _update_state_summary(widget)
        _say(
            widget,
            f"prepared {len(rows)} curated droplet target(s) for Raman MDA; "
            "review the MDA summary before Run",
        )
    except Exception as exc:
        widget._dry["mda_prepared"] = False
        _update_state_summary(widget)
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
            "raw_displacement_px", "registered_distance_px",
            "tracking_confidence", "field_shift_y_px", "field_shift_x_px",
            "adaptive_class",
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
                        t.get("raw_displacement_px", ""),
                        t.get("registered_distance_px", t.get("match_distance_px", "")),
                        t.get("tracking_confidence", ""),
                        (widget._dry.get("field_shift_yx") or ["", ""])[0]
                        if not isinstance(widget._dry.get("field_shift_yx"), np.ndarray)
                        else widget._dry["field_shift_yx"][0],
                        (widget._dry.get("field_shift_yx") or ["", ""])[1]
                        if not isinstance(widget._dry.get("field_shift_yx"), np.ndarray)
                        else widget._dry["field_shift_yx"][1],
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
