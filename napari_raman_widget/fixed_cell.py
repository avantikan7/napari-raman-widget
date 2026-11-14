"""Fixed-cell Z-stack lipid-droplet targeting for the shared Raman MDA.

Jeff's MDA remains authoritative for Time, Positions, stage/camera Grid,
Z Stack, Channels, Raman selection, Autofocus and Run. This panel analyzes an
acquired BODIPY/CellMask stack and prepares objective-calibrated stage targets.
Analysis and validation never move hardware.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QPushButton, QSpinBox, QTableWidget,
    QTableWidgetItem, QVBoxLayout,
)

from .ui_helpers import make_collapsible

STACK = "fixed-cell BODIPY Z-stack"
BEST = "fixed-cell global focus preview"
RAW = "fixed-cell LD labels (3D)"
FOCUSED = "fixed-cell focused LD masks"
AIMS = "fixed-cell LD centers"
GRID = "fixed-cell intra-LD Raman grid"
CELLMASK = "fixed-cell CellMask Orange"
CELLS = "fixed-cell whole-cell masks"


@dataclass(frozen=True)
class PixelScale:
    x_um: float
    y_um: float
    objective: str | None = None
    source: str = "manual"

    @property
    def mean_um(self):
        return math.sqrt(float(self.x_um) * float(self.y_um))


def diameter_um_to_area_px(diameter_um, scale):
    if diameter_um <= 0 or scale.x_um <= 0 or scale.y_um <= 0:
        raise ValueError("diameter and pixel scale must be positive")
    return math.pi * (float(diameter_um) / 2.0) ** 2 / (
        float(scale.x_um) * float(scale.y_um)
    )


def area_px_to_diameter_um(area_px, scale):
    if area_px < 0 or scale.x_um <= 0 or scale.y_um <= 0:
        raise ValueError("area must be non-negative and scale positive")
    area_um2 = float(area_px) * float(scale.x_um) * float(scale.y_um)
    return 2.0 * math.sqrt(area_um2 / math.pi)


def tenengrad_score(image, mask=None):
    """Mean squared Sobel gradient; larger values indicate sharper structure."""
    from scipy import ndimage as ndi

    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("focus scoring expects a 2D plane")
    gy = ndi.sobel(image, axis=0, mode="nearest")
    gx = ndi.sobel(image, axis=1, mode="nearest")
    energy = gx * gx + gy * gy
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != image.shape or not mask.any():
            return 0.0
        mask = ndi.binary_dilation(mask, iterations=2)
        energy = energy[mask]
    return float(energy.mean()) if energy.size else 0.0


def focus_profile(stack):
    stack = np.asarray(stack)
    if stack.ndim != 3:
        raise ValueError("focus profile expects Z-Y-X data")
    return np.asarray([tenengrad_score(plane) for plane in stack], float)


def cellpose_volume_kwargs(
    mode, *, z_step_um, scale, cellprob, flow, stitch,
    anisotropy_override=0.0,
):
    """Explicit Cellpose settings for 2D+stitch or 2.5D/3D flow."""
    if mode not in {"stitch", "3d"}:
        raise ValueError(f"unknown Z segmentation mode: {mode}")
    kwargs = dict(
        diameter=None,
        flow_threshold=float(flow),
        cellprob_threshold=float(cellprob),
        normalize=True,
        z_axis=0,
        channel_axis=None,
        min_size=1,
    )
    if mode == "3d":
        anisotropy = (
            float(anisotropy_override)
            if float(anisotropy_override) > 0
            else float(z_step_um) / float(scale.mean_um)
        )
        kwargs.update(
            do_3D=True,
            anisotropy=max(anisotropy, 0.01),
            stitch_threshold=0.0,
        )
    else:
        kwargs.update(do_3D=False, stitch_threshold=float(stitch))
    return kwargs


def _deepest_point(mask):
    from scipy import ndimage as ndi

    distance = ndi.distance_transform_edt(np.asarray(mask, bool))
    if not np.any(distance > 0):
        raise ValueError("mask has no interior")
    y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
    return (int(y), int(x)), float(distance[y, x])


def _grid_inside(mask, aim, spacing_px, margin_px):
    from scipy import ndimage as ndi

    mask = np.asarray(mask, bool)
    safe = ndi.distance_transform_edt(mask) >= max(0.0, float(margin_px))
    ys, xs = np.where(safe)
    if not len(ys):
        return np.asarray([aim], float)
    spacing = max(1, int(round(spacing_px)))
    y0, x0 = map(int, aim)
    keep = (((ys - y0) % spacing == 0) & ((xs - x0) % spacing == 0))
    points = np.column_stack([ys[keep], xs[keep]]).astype(float)
    if not len(points):
        points = np.asarray([aim], float)
    elif not np.any(np.all(points == np.asarray(aim), axis=1)):
        points = np.vstack([np.asarray(aim, float), points])
    return points


def focused_droplet_targets(
    stack,
    labels,
    *,
    scale,
    min_diameter_um,
    max_diameter_um,
    min_focus_confidence,
    min_z_span,
    max_z_span,
    reject_edge_focus,
    grid_spacing_um,
    grid_margin_um,
):
    """Reduce 3D/stitched labels to one best-focus Raman target per LD."""
    stack = np.asarray(stack, np.float32)
    labels = np.asarray(labels)
    if stack.ndim != 3 or labels.shape != stack.shape:
        raise ValueError("stack and labels must share a Z-Y-X shape")
    if min_diameter_um <= 0 or max_diameter_um <= min_diameter_um:
        raise ValueError("physical LD diameter limits are invalid")

    focused = np.zeros(labels.shape, np.uint32)
    targets = []
    next_label = 1
    for source_label in np.unique(labels):
        if source_label == 0:
            continue
        z_indices = np.flatnonzero(np.any(labels == source_label, axis=(1, 2)))
        z_span = int(len(z_indices))
        if z_span < int(min_z_span):
            continue
        if int(max_z_span) > 0 and z_span > int(max_z_span):
            continue

        scores = [
            tenengrad_score(stack[z], labels[z] == source_label)
            for z in z_indices
        ]
        order = np.argsort(scores)[::-1]
        best_local = int(order[0])
        best_z = int(z_indices[best_local])
        best_score = float(scores[best_local])
        second = float(scores[int(order[1])]) if len(order) > 1 else 0.0
        confidence = (
            (best_score - second) / max(abs(best_score), np.finfo(float).eps)
            if len(order) > 1 else 1.0
        )
        if reject_edge_focus and stack.shape[0] > 1 and best_z in (
            0, stack.shape[0] - 1,
        ):
            continue
        if confidence < float(min_focus_confidence):
            continue

        mask = labels[best_z] == source_label
        area_px = int(mask.sum())
        diameter_um = area_px_to_diameter_um(area_px, scale)
        if not (float(min_diameter_um) <= diameter_um <= float(max_diameter_um)):
            continue
        try:
            aim, radius_px = _deepest_point(mask)
        except ValueError:
            continue
        grid = _grid_inside(
            mask,
            aim,
            float(grid_spacing_um) / scale.mean_um,
            float(grid_margin_um) / scale.mean_um,
        )
        focused[best_z][mask] = next_label
        targets.append(dict(
            id=next_label,
            source_label=int(source_label),
            z=best_z,
            aim=aim,
            area_px=area_px,
            diameter_um=float(diameter_um),
            radius_px=float(radius_px),
            focus_score=best_score,
            focus_confidence=float(confidence),
            z_span=z_span,
            grid=grid,
            cell_id=0,
            selected=True,
        ))
        next_label += 1
    return focused, targets


def assign_targets_to_cells(targets, cell_labels, *, require_cell=True, max_per_cell=0):
    """Assign LD centers to whole-cell labels and enforce a per-cell limit."""
    cell_labels = np.asarray(cell_labels)
    if cell_labels.ndim != 2:
        raise ValueError("whole-cell labels must be 2D")
    grouped = defaultdict(list)
    for target in targets:
        y, x = map(int, target["aim"])
        cell_id = (
            int(cell_labels[y, x])
            if 0 <= y < cell_labels.shape[0] and 0 <= x < cell_labels.shape[1]
            else 0
        )
        target["cell_id"] = cell_id
        target["selected"] = not require_cell or cell_id > 0
        if target["selected"]:
            grouped[cell_id].append(target)
    if int(max_per_cell) > 0:
        for rows in grouped.values():
            rows.sort(
                key=lambda row: (
                    row.get("focus_confidence", 0.0),
                    row.get("radius_px", 0.0),
                ),
                reverse=True,
            )
            for row in rows[int(max_per_cell):]:
                row["selected"] = False
    return targets


def _update_fixed_ui_state(owner):
    """Progressively enable the fixed-cell workflow without touching hardware."""
    fixed = getattr(owner, "_fixed", {}) or {}
    has_stack = fixed.get("stack") is not None
    has_cellmask = fixed.get("cellmask_image") is not None
    has_targets = bool(_selected_targets(owner)) if fixed.get("targets") else False
    has_positions = bool(fixed.get("stage_positions"))

    for name, enabled in (
        ("fixed_focus_btn", has_stack),
        ("fixed_segment_btn", has_stack),
        ("fixed_segment_cells_btn", has_cellmask),
        ("fixed_validate_btn", has_targets),
        ("fixed_prepare_btn", has_positions),
    ):
        control = getattr(owner, name, None)
        if control is not None:
            control.setEnabled(enabled)

    readiness = getattr(owner, "fixed_readiness", None)
    if readiness is not None:
        cell_count = (
            len({row.get("cell_id", 0) for row in _selected_targets(owner)
                 if row.get("cell_id", 0) > 0})
            if has_targets else 0
        )
        readiness.setText(
            f"{'✓' if has_stack else '○'} Stack loaded   "
            f"{'✓' if has_targets else '○'} Focused LDs: "
            f"{len(_selected_targets(owner)) if has_targets else 0}\n"
            f"{'✓' if cell_count else '○'} Cells assigned: {cell_count}   "
            f"{'✓' if has_positions else '○'} Stage targets validated\n"
            f"{'✓' if fixed.get('mda_prepared') else '○'} MDA prepared"
        )

    try:
        from .workflow_docks import refresh_control_hub
        refresh_control_hub(owner)
    except Exception:
        pass


def _say(owner, message):
    owner.fixed_status.setText(message)
    owner.status.setText(f"Status: {message}")
    _update_fixed_ui_state(owner)


def _remove(owner, name):
    if name in owner.viewer.layers:
        owner.viewer.layers.remove(name)


def _show_shared_mda(owner):
    if owner._unified_mda is None or owner.main_window is None:
        _say(owner, "Connect hardware first to open Jeff's shared MDA")
        return
    dock = owner.main_window._dock_widgets.get("MDA")
    if dock is not None:
        dock.show()
        dock.raise_()
    _refresh_mda_summary(owner)
    _say(owner, "Opened Jeff's shared MDA setup")


def _refresh_mda_summary(owner):
    if owner._unified_mda is None:
        owner.fixed_mda_summary.setText(
            "MDA not connected. Connect hardware, then configure Channels, "
            "Z Stack, Positions/Grid and optional Time."
        )
        return
    try:
        sequence = owner._unified_mda.value()
        channels = ", ".join(c.config for c in sequence.channels) or "none"
        if sequence.z_plan is None:
            z_text = "off"
        else:
            values = list(sequence.z_plan.positions())
            step = getattr(sequence.z_plan, "step", None)
            z_text = (
                f"{len(values)} planes; {min(values):g}..{max(values):g} µm"
                + (f"; step={float(step):g} µm" if step else "")
                if values else "configured"
            )
        owner.fixed_mda_summary.setText(
            f"Shared MDA: channels={channels} | Z={z_text} | "
            f"positions={len(sequence.stage_positions)} | "
            f"field grid={'on' if sequence.grid_plan is not None else 'off'} | "
            f"time={'on' if sequence.time_plan is not None else 'off'}"
        )
    except Exception as error:
        owner.fixed_mda_summary.setText(f"Could not read MDA plan: {error}")


def _objective_scale(owner):
    fallback = float(owner.fixed_pixel_input.value())
    objective = owner._current_objective() if owner.core is not None else None
    if owner.fixed_auto_scale_check.isChecked() and owner.core is not None:
        try:
            current = owner._ensure_current_vandermonde()
            if current and owner.vandermonde is not None:
                from .grid_calibration import vandermonde_fov
                width, height = owner._get_image_xy()
                fov_x, fov_y = vandermonde_fov(owner.vandermonde, width, height)
                scale = PixelScale(
                    float(fov_x) / float(width),
                    float(fov_y) / float(height),
                    str(current),
                    "objective-specific Vandermonde calibration",
                )
                owner._fixed["pixel_scale"] = scale
                owner.fixed_scale_status.setText(
                    f"Objective index {current}: X={scale.x_um:.4f}, "
                    f"Y={scale.y_um:.4f} µm/px; camera binning must match."
                )
                _update_physical_summary(owner)
                return scale
        except Exception as error:
            owner.fixed_scale_status.setText(
                f"Objective calibration unavailable ({error}); using fallback."
            )
    scale = PixelScale(fallback, fallback, objective, "manual fallback")
    owner._fixed["pixel_scale"] = scale
    owner.fixed_scale_status.setText(
        f"Fallback scale: {fallback:.4f} µm/px"
        + (f" for objective index {objective}" if objective else "")
    )
    _update_physical_summary(owner)
    return scale


def _update_physical_summary(owner):
    scale = owner._fixed.get("pixel_scale") or PixelScale(
        float(owner.fixed_pixel_input.value()),
        float(owner.fixed_pixel_input.value()),
    )
    try:
        minimum = float(owner.fixed_min_diam_input.value())
        maximum = float(owner.fixed_max_diam_input.value())
        amin = diameter_um_to_area_px(minimum, scale)
        amax = diameter_um_to_area_px(maximum, scale)
        owner.fixed_physical_summary.setText(
            f"{minimum:g}–{maximum:g} µm = {amin:.1f}–{amax:.1f} px² "
            f"at X={scale.x_um:.4f}, Y={scale.y_um:.4f} µm/px."
        )
    except Exception as error:
        owner.fixed_physical_summary.setText(str(error))


def _axis_labels(owner, layer, data_ndim):
    labels = tuple(str(v).lower() for v in owner.viewer.dims.axis_labels)
    spatial_ndim = data_ndim - 1 if getattr(layer, "rgb", False) else data_ndim
    if len(labels) >= spatial_ndim:
        return labels[-spatial_ndim:]
    return ("t", "p", "g", "c", "z", "y", "x")[-spatial_ndim:]


def _extract_zyx(owner, layer):
    data = layer.data
    rgb = bool(getattr(layer, "rgb", False))
    labels = _axis_labels(owner, layer, len(data.shape))
    spatial_shape = data.shape[:-1] if rgb else data.shape
    if "y" not in labels or "x" not in labels:
        raise ValueError(f"Could not identify Y/X axes from {labels}")
    steps = list(owner.viewer.dims.current_step)
    if len(steps) >= len(labels):
        steps = steps[-len(labels):]
    else:
        steps = [0] * (len(labels) - len(steps)) + steps
    index, retained = [], []
    context = {"t": 0, "p": 0, "g": 0, "c": 0}
    for axis, label in enumerate(labels):
        if label in {"y", "x", "z"}:
            index.append(slice(None))
            retained.append(label)
        else:
            value = min(max(int(steps[axis]), 0), int(spatial_shape[axis]) - 1)
            index.append(value)
            if label in context:
                context[label] = value
    if rgb:
        index.append(0)
    array = np.asarray(data[tuple(index)], np.float32)
    if "z" not in retained:
        if array.ndim != 2:
            raise ValueError(f"Selected slice is not 2D: {array.shape}")
        array = array[None]
    else:
        if array.ndim != 3:
            raise ValueError(f"Selected stack is not 3D: {array.shape}")
        order = [retained.index(axis) for axis in ("z", "y", "x")]
        array = np.transpose(array, order)
    sequence = getattr(layer, "metadata", {}).get("useq_sequence")
    if sequence is not None:
        from useq import MDASequence
        if isinstance(sequence, str):
            sequence = MDASequence.model_validate_json(sequence)
        elif isinstance(sequence, dict):
            sequence = MDASequence.model_validate(sequence)
    return array, context, sequence


def _use_bodipy(owner):
    from napari.layers import Image
    layer = owner.viewer.layers.selection.active
    if not isinstance(layer, Image) or layer.name in {STACK, BEST, CELLMASK}:
        _say(owner, "Select the acquired BODIPY image layer first")
        return
    try:
        stack, context, sequence = _extract_zyx(owner, layer)
    except Exception as error:
        _say(owner, f"Could not use BODIPY layer: {error}")
        return
    for name in (STACK, BEST, RAW, FOCUSED, AIMS, GRID):
        _remove(owner, name)
    owner.viewer.add_image(
        stack, name=STACK, colormap="green", blending="additive",
        contrast_limits=[float(v) for v in np.percentile(stack, [1, 99.8])],
    )
    owner._fixed.update(
        stack=stack, source_layer=layer, source_context=context,
        source_sequence=sequence, targets=[], rejected=set(),
        stage_positions=[], mda_prepared=False,
    )
    _objective_scale(owner)
    _score_global_focus(owner)
    _say(owner, f"Loaded BODIPY Z-stack: {stack.shape[0]} plane(s), "
                f"{stack.shape[1]}×{stack.shape[2]} px.")


def _score_global_focus(owner):
    stack = owner._fixed.get("stack")
    if stack is None:
        _say(owner, "Load a BODIPY Z-stack first")
        return
    scores = focus_profile(stack)
    best = int(np.argmax(scores))
    owner._fixed["global_focus_scores"] = scores
    owner._fixed["global_best_z"] = best
    _remove(owner, BEST)
    owner.viewer.add_image(
        stack[best], name=BEST, colormap="green", blending="additive"
    )
    owner.fixed_focus_status.setText(
        f"Global preview: Z index {best}, score={scores[best]:.4g}. "
        "Final Raman Z is selected separately for each stitched LD."
    )


def _source_z_step(owner):
    sequence = owner._fixed.get("source_sequence")
    plan = getattr(sequence, "z_plan", None) if sequence is not None else None
    step = getattr(plan, "step", None)
    if step is not None and np.isfinite(step) and step > 0:
        owner.fixed_z_status.setText(f"Z step from source MDA: {float(step):g} µm")
        return float(step)
    fallback = float(owner.fixed_z_step_input.value())
    owner.fixed_z_status.setText(
        f"Z step fallback: {fallback:g} µm (source metadata unavailable)"
    )
    return fallback


def _segment_droplets(owner):
    from napari.qt.threading import thread_worker
    stack = owner._fixed.get("stack")
    if stack is None:
        _say(owner, "Load a BODIPY Z-stack first")
        return
    try:
        model_name = owner._selected_cellpose_model(owner.fixed_ld_model_combo)
    except Exception as error:
        _say(owner, str(error))
        return
    scale = _objective_scale(owner)
    kwargs = cellpose_volume_kwargs(
        owner.fixed_z_mode.currentData(),
        z_step_um=_source_z_step(owner),
        scale=scale,
        cellprob=float(owner.fixed_cellprob_input.value()),
        flow=float(owner.fixed_flow_input.value()),
        stitch=float(owner.fixed_stitch_input.value()),
        anisotropy_override=float(owner.fixed_anisotropy_input.value()),
    )
    target_params = dict(
        scale=scale,
        min_diameter_um=float(owner.fixed_min_diam_input.value()),
        max_diameter_um=float(owner.fixed_max_diam_input.value()),
        min_focus_confidence=float(owner.fixed_focus_conf_input.value()),
        min_z_span=int(owner.fixed_min_z_span_input.value()),
        max_z_span=int(owner.fixed_max_z_span_input.value()),
        reject_edge_focus=owner.fixed_reject_edge_check.isChecked(),
        grid_spacing_um=float(owner.fixed_grid_spacing_input.value()),
        grid_margin_um=float(owner.fixed_grid_margin_input.value()),
    )
    owner.fixed_segment_btn.setEnabled(False)
    _say(owner, f"Loading Cellpose model {model_name}…")

    @thread_worker
    def work():
        from raman_mda_engine.aiming._cellpose import (
            eval_cellpose_model, new_cellpose_model,
        )
        model = new_cellpose_model(model_name)
        masks = np.asarray(eval_cellpose_model(model, stack, **kwargs))
        if masks.ndim == 2:
            volume = np.zeros(stack.shape, np.uint32)
            volume[int(owner._fixed.get("global_best_z", 0))] = masks
            masks = volume
        if masks.shape != stack.shape:
            raise ValueError(f"Cellpose returned {masks.shape}; expected {stack.shape}")
        focused, targets = focused_droplet_targets(
            stack, masks, **target_params
        )
        return masks.astype(np.uint32), focused, targets, kwargs

    def finished(result):
        masks, focused, targets, used_kwargs = result
        owner._fixed.update(
            raw_labels=masks, focused_labels=focused, targets=targets,
            rejected=set(), stage_positions=[], mda_prepared=False,
            cellpose_kwargs=used_kwargs,
        )
        for name in (RAW, FOCUSED, AIMS, GRID):
            _remove(owner, name)
        owner.viewer.add_labels(masks, name=RAW, opacity=0.18)
        owner.viewer.add_labels(focused, name=FOCUSED, opacity=0.55)
        owner.viewer.add_points(
            np.asarray([[row["z"], *row["aim"]] for row in targets], float).reshape((-1, 3)),
            name=AIMS, ndim=3, size=9, face_color="red",
        )
        _assign_cells(owner)
        owner.fixed_segment_btn.setEnabled(True)
        _say(owner, f"Focused LD segmentation: {len(targets)} target(s) from "
                    f"{len(np.unique(masks)) - 1} Cellpose object(s).")

    def failed(error):
        owner.fixed_segment_btn.setEnabled(True)
        _say(owner, f"Focused LD segmentation failed: {error}")

    worker = work()
    worker.returned.connect(finished)
    worker.errored.connect(failed)
    worker.start()
    owner._fixed_worker = worker


def _use_cellmask(owner):
    from napari.layers import Image
    layer = owner.viewer.layers.selection.active
    if not isinstance(layer, Image) or layer.name in {STACK, BEST, CELLMASK}:
        _say(owner, "Select the acquired CellMask Orange layer first")
        return
    try:
        stack, _context, _sequence = _extract_zyx(owner, layer)
    except Exception as error:
        _say(owner, f"Could not use CellMask Orange: {error}")
        return
    bodipy = owner._fixed.get("stack")
    if bodipy is not None and stack.shape[-2:] != bodipy.shape[-2:]:
        _say(owner, "CellMask and BODIPY image dimensions do not match")
        return
    projection = np.max(stack, axis=0)
    _remove(owner, CELLMASK)
    owner.viewer.add_image(
        projection, name=CELLMASK, colormap="orange", blending="additive",
        contrast_limits=[float(v) for v in np.percentile(projection, [1, 99.8])],
    )
    owner._fixed["cellmask_image"] = projection
    _say(owner, f"Loaded CellMask Orange projection: {projection.shape}")


def _segment_cells(owner):
    from napari.qt.threading import thread_worker
    image = owner._fixed.get("cellmask_image")
    if image is None:
        _say(owner, "Load the CellMask Orange layer first")
        return
    try:
        model_name = owner._selected_cellpose_model(owner.fixed_cell_model_combo)
    except Exception as error:
        _say(owner, str(error))
        return
    scale = _objective_scale(owner)
    diameter_px = float(owner.fixed_cell_diam_input.value()) / scale.mean_um
    owner.fixed_segment_cells_btn.setEnabled(False)

    @thread_worker
    def work():
        from raman_mda_engine.aiming._cellpose import (
            eval_cellpose_model, new_cellpose_model,
        )
        model = new_cellpose_model(model_name)
        return np.asarray(
            eval_cellpose_model(
                model,
                image,
                diameter=diameter_px,
                flow_threshold=float(owner.fixed_cell_flow_input.value()),
                cellprob_threshold=float(owner.fixed_cell_prob_input.value()),
                normalize=True,
                do_3D=False,
                channel_axis=None,
                min_size=25,
            ),
            dtype=np.uint32,
        )

    def finished(labels):
        owner._fixed["cell_labels"] = labels
        _remove(owner, CELLS)
        owner.viewer.add_labels(labels, name=CELLS, opacity=0.35)
        _assign_cells(owner)
        owner.fixed_segment_cells_btn.setEnabled(True)
        _say(owner, f"Segmented {len(np.unique(labels)) - 1} whole cell(s).")

    def failed(error):
        owner.fixed_segment_cells_btn.setEnabled(True)
        _say(owner, f"CellMask segmentation failed: {error}")

    worker = work()
    worker.returned.connect(finished)
    worker.errored.connect(failed)
    worker.start()
    owner._fixed_cell_worker = worker


def _assign_cells(owner):
    targets = owner._fixed.get("targets", [])
    labels = owner._fixed.get("cell_labels")
    if labels is not None:
        assign_targets_to_cells(
            targets,
            labels,
            require_cell=owner.fixed_require_cell_check.isChecked(),
            max_per_cell=int(owner.fixed_max_per_cell_input.value()),
        )
    owner._fixed["stage_positions"] = []
    owner._fixed["mda_prepared"] = False
    _refresh_targets(owner)


def _selected_targets(owner):
    rejected = set(owner._fixed.get("rejected", set()))
    return [
        row for row in owner._fixed.get("targets", [])
        if row.get("selected", True) and row["id"] not in rejected
    ]


def _refresh_targets(owner):
    targets = _selected_targets(owner)
    if AIMS in owner.viewer.layers:
        owner.viewer.layers[AIMS].data = np.asarray(
            [[row["z"], *row["aim"]] for row in targets], float
        ).reshape((-1, 3))
    grid_points = []
    if owner.fixed_intra_grid_check.isChecked():
        for row in targets:
            grid_points.extend([[row["z"], *point] for point in row["grid"]])
    if GRID in owner.viewer.layers:
        owner.viewer.layers[GRID].data = np.asarray(grid_points, float).reshape((-1, 3))
    elif grid_points:
        owner.viewer.add_points(
            np.asarray(grid_points, float).reshape((-1, 3)),
            name=GRID, ndim=3, size=3, face_color="yellow",
        )

    all_targets = owner._fixed.get("targets", [])
    selected_ids = {row["id"] for row in targets}
    owner.fixed_target_table.setRowCount(len(all_targets))
    for r, target in enumerate(all_targets):
        values = (
            target["id"], target.get("cell_id", 0), target["z"],
            f"{target['diameter_um']:.2f}",
            f"{target['focus_confidence']:.3f}", target["z_span"],
            "yes" if target["id"] in selected_ids else "no",
        )
        for c, value in enumerate(values):
            item = QTableWidgetItem(str(value))
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            owner.fixed_target_table.setItem(r, c, item)
    assigned = {row.get("cell_id", 0) for row in targets if row.get("cell_id", 0) > 0}
    mode = (
        "stage-centered intra-LD Raman grid"
        if owner.fixed_intra_grid_check.isChecked()
        else "one stage-centered Raman point per LD"
    )
    owner.fixed_target_summary.setText(
        f"Selected {len(targets)} focused LD(s) across {len(assigned)} "
        f"CellMask cell(s); {mode}."
    )
    _update_fixed_ui_state(owner)


def _reject_selected(owner):
    if AIMS not in owner.viewer.layers:
        return
    visible = _selected_targets(owner)
    indices = sorted(getattr(owner.viewer.layers[AIMS], "selected_data", set()))
    rejected = set(owner._fixed.get("rejected", set()))
    for index in indices:
        if 0 <= index < len(visible):
            rejected.add(visible[index]["id"])
    owner._fixed["rejected"] = rejected
    _refresh_targets(owner)


def _target_rows(owner, target, use_grid):
    context = owner._fixed.get("source_context", {})
    points = target["grid"] if use_grid else np.asarray([target["aim"]])
    return np.asarray([
        [
            int(context.get("t", 0)), int(context.get("p", 0)),
            int(context.get("g", 0)), int(context.get("c", 0)),
            int(target["z"]), float(point[0]), float(point[1]),
        ]
        for point in points
    ], float)


def _build_stage_positions(owner):
    sequence = owner._fixed.get("source_sequence")
    targets = _selected_targets(owner)
    if sequence is None:
        raise ValueError(
            "Selected stack has no useq_sequence metadata. Acquire it through "
            "Jeff's MDA so stage XYZ and Z-plane provenance are present."
        )
    if not targets:
        raise ValueError("No focused LD targets are selected")
    objective = owner._ensure_current_vandermonde()
    if not objective or owner.vandermonde is None:
        raise ValueError("No pixel-to-stage calibration for the active objective")
    from .raman_selection import selected_positions
    shape = tuple(owner._fixed["stack"].shape[-2:])
    use_grid = owner.fixed_intra_grid_check.isChecked()
    positions = []
    for target in targets:
        generated = selected_positions(
            sequence,
            _target_rows(owner, target, use_grid),
            shape,
            owner.vandermonde,
            batch=use_grid,
            provenance=True,
            dataset=getattr(owner._fixed.get("source_layer"), "name", None),
            batch_center=(np.asarray(target["aim"], float) if use_grid else None),
        )
        if len(generated) != 1:
            raise ValueError(f"Expected one stage position for LD {target['id']}")
        position = generated[0]
        sub = position.sequence
        metadata = deepcopy(sub.metadata)
        metadata["lipid_droplet"] = dict(
            ld_id=int(target["id"]),
            cell_id=int(target.get("cell_id", 0)),
            best_z_index=int(target["z"]),
            diameter_um=float(target["diameter_um"]),
            focus_confidence=float(target["focus_confidence"]),
            objective_index=str(objective),
        )
        positions.append(position.replace(
            name=f"Cell {target.get('cell_id', 0)} LD {target['id']}",
            sequence=sub.replace(metadata=metadata),
        ))
    return positions


def _validate_stage(owner):
    try:
        positions = _build_stage_positions(owner)
    except Exception as error:
        owner._fixed["stage_positions"] = []
        owner.fixed_stage_status.setText(f"Validation failed: {error}")
        _say(owner, f"Pixel→stage validation failed: {error}")
        return False
    xyz = np.asarray([[p.x, p.y, p.z] for p in positions], float)
    if not np.isfinite(xyz).all():
        _say(owner, "Pixel→stage validation returned non-finite XYZ")
        return False
    owner._fixed["stage_positions"] = positions
    owner.fixed_stage_status.setText(
        f"Validated {len(positions)} target(s): "
        f"X={xyz[:,0].min():.2f}..{xyz[:,0].max():.2f} µm; "
        f"Y={xyz[:,1].min():.2f}..{xyz[:,1].max():.2f} µm. "
        "No stage, galvo or laser moved."
    )
    _say(owner, "Pixel→stage validation passed; no hardware moved")
    return True


def _prepare_mda(owner):
    if not owner._fixed.get("stage_positions") and not _validate_stage(owner):
        return
    try:
        from useq import MDASequence
        from .unified_mda import apply_prepared_positions
        positions = owner._fixed["stage_positions"]
        owner.selection_results = dict(
            selection_type="fixed_cell_droplets",
            sources=[],
            autofocus_p=np.arange(len(positions)),
            new_seq=MDASequence(stage_positions=positions),
            batch=owner.fixed_intra_grid_check.isChecked(),
            pre_acq=owner.fixed_intra_grid_check.isChecked(),
        )
        apply_prepared_positions(owner)
        owner._fixed["mda_prepared"] = True
        _say(owner, f"Prepared {len(positions)} focused LD stage position(s) "
                    "in Jeff's shared MDA. Review MDA before Run.")
    except Exception as error:
        owner._fixed["mda_prepared"] = False
        _say(owner, f"Could not prepare stage-position MDA: {error}")


def build_fixed_cell_controls(owner):
    """Build the four-step fixed-cell validation workflow."""
    from .widget import _cellpose_model_names

    box = make_collapsible("Fixed-cell Z-stack Targeting", expanded=True)
    layout = QVBoxLayout(box)

    intro = QLabel(
        "Fixed-cell validation first: acquire through Jeff's shared MDA, "
        "segment focused LDs, review them per cell, then prepare stage targets."
    )
    intro.setWordWrap(True)
    intro.setToolTip(
        "This workflow adds biological target selection only. It does not "
        "replace Jeff's Time, Positions, Grid, Z Stack, Channels, Raman "
        "selection, Autofocus, Save, Run or Cancel controls."
    )
    layout.addWidget(intro)

    mode_row = QHBoxLayout()
    mode_row.addWidget(QLabel("Interface:"))
    owner.fixed_interface_mode = QComboBox()
    owner.fixed_interface_mode.addItems(["Basic", "Expert"])
    owner.fixed_interface_mode.setToolTip(
        "Basic shows the decisions needed for routine use. Expert reveals "
        "Cellpose thresholds, anisotropy, focus-confidence and Z-span filters."
    )
    mode_row.addWidget(owner.fixed_interface_mode)
    layout.addLayout(mode_row)

    owner.fixed_readiness = QLabel()
    owner.fixed_readiness.setWordWrap(True)
    owner.fixed_readiness.setStyleSheet(
        "QLabel { border: 1px solid palette(mid); padding: 6px; "
        "font-family: monospace; }"
    )
    owner.fixed_readiness.setToolTip(
        "Read-only readiness summary. Checkmarks indicate completed software "
        "steps; no hardware is queried or moved."
    )
    layout.addWidget(owner.fixed_readiness)

    # ---------------- 1. ACQUIRE ----------------
    acquire = QGroupBox("1 — Acquire")
    acquire_layout = QVBoxLayout(acquire)
    acquire_note = QLabel(
        "Configure BODIPY, CellMask Orange, Z Stack, Positions or field Grid "
        "in the unchanged shared MDA, then acquire the fixed-cell data."
    )
    acquire_note.setWordWrap(True)
    acquire_layout.addWidget(acquire_note)

    owner.fixed_open_mda_btn = QPushButton("Open shared MDA setup")
    owner.fixed_open_mda_btn.setToolTip(
        "Open Jeff's authoritative MDA window. This button only reveals the "
        "existing dock and does not move hardware."
    )
    owner.fixed_open_mda_btn.clicked.connect(
        lambda _checked=False: _show_shared_mda(owner)
    )
    acquire_layout.addWidget(owner.fixed_open_mda_btn)

    owner.fixed_refresh_mda_btn = QPushButton("Refresh acquisition summary")
    owner.fixed_refresh_mda_btn.setToolTip(
        "Read the configured channels, Z plan, positions, field grid and time "
        "plan. No acquisition or stage movement occurs."
    )
    owner.fixed_refresh_mda_btn.clicked.connect(
        lambda _checked=False: _refresh_mda_summary(owner)
    )
    acquire_layout.addWidget(owner.fixed_refresh_mda_btn)

    owner.fixed_mda_summary = QLabel(
        "MDA not connected. Connect hardware to configure acquisition."
    )
    owner.fixed_mda_summary.setWordWrap(True)
    acquire_layout.addWidget(owner.fixed_mda_summary)
    layout.addWidget(acquire)

    # ---------------- 2. SEGMENT ----------------
    segment_step = QGroupBox("2 — Segment")
    segment_layout = QVBoxLayout(segment_step)

    scale_group = QGroupBox("Objective calibration and physical LD size")
    scale_form = QFormLayout(scale_group)
    owner.fixed_auto_scale_check = QCheckBox(
        "Use active objective's pixel-to-stage calibration"
    )
    owner.fixed_auto_scale_check.setChecked(True)
    owner.fixed_auto_scale_check.setToolTip(
        "Recommended. Read the current IX83 objective state and its "
        "Vandermonde model, then calculate X/Y µm per pixel. The turret is not moved."
    )
    scale_form.addRow(owner.fixed_auto_scale_check)

    owner.fixed_pixel_input = QDoubleSpinBox()
    owner.fixed_pixel_input.setRange(0.001, 20.0)
    owner.fixed_pixel_input.setDecimals(5)
    owner.fixed_pixel_input.setValue(0.325)
    owner.fixed_pixel_input.setSuffix(" µm/px")
    owner.fixed_pixel_input.setToolTip(
        "Manual fallback only. Koseki's current 40× estimate is 0.325 µm/px."
    )
    scale_form.addRow("Fallback pixel size", owner.fixed_pixel_input)

    owner.fixed_scale_btn = QPushButton("Read active objective calibration")
    owner.fixed_scale_btn.setToolTip(
        "Read the active objective index and calculate X/Y µm-per-pixel. "
        "No objective or stage motion."
    )
    owner.fixed_scale_btn.clicked.connect(
        lambda _checked=False: _objective_scale(owner)
    )
    scale_form.addRow(owner.fixed_scale_btn)

    owner.fixed_scale_status = QLabel("Objective scale not read yet.")
    owner.fixed_scale_status.setWordWrap(True)
    scale_form.addRow(owner.fixed_scale_status)

    owner.fixed_min_diam_input = QDoubleSpinBox()
    owner.fixed_min_diam_input.setRange(0.1, 100.0)
    owner.fixed_min_diam_input.setValue(1.0)
    owner.fixed_min_diam_input.setSuffix(" µm")
    owner.fixed_min_diam_input.setToolTip(
        "Minimum equivalent LD diameter on its selected best-focus plane. "
        "At 0.325 µm/px, 1 µm is ≈3.08 px diameter and ≈7.4 px²."
    )
    scale_form.addRow("Minimum LD diameter", owner.fixed_min_diam_input)

    owner.fixed_max_diam_input = QDoubleSpinBox()
    owner.fixed_max_diam_input.setRange(0.2, 100.0)
    owner.fixed_max_diam_input.setValue(10.0)
    owner.fixed_max_diam_input.setSuffix(" µm")
    owner.fixed_max_diam_input.setToolTip(
        "Editable conservative upper limit. It replaces the old 3000-pixel maximum."
    )
    scale_form.addRow("Maximum LD diameter", owner.fixed_max_diam_input)

    owner.fixed_physical_summary = QLabel()
    owner.fixed_physical_summary.setWordWrap(True)
    scale_form.addRow(owner.fixed_physical_summary)
    for control in (
        owner.fixed_pixel_input,
        owner.fixed_min_diam_input,
        owner.fixed_max_diam_input,
    ):
        control.valueChanged.connect(
            lambda _value=0: _update_physical_summary(owner)
        )
    segment_layout.addWidget(scale_group)

    ld_group = QGroupBox("BODIPY Z-stack and focused LD segmentation")
    ld_layout = QVBoxLayout(ld_group)
    owner.fixed_use_bodipy_btn = QPushButton(
        "Use selected acquired layer as BODIPY Z-stack"
    )
    owner.fixed_use_bodipy_btn.setToolTip(
        "Select a grayscale MDA image layer. Current t/p/g/c values are held "
        "and all Z planes are extracted."
    )
    owner.fixed_use_bodipy_btn.clicked.connect(
        lambda _checked=False: _use_bodipy(owner)
    )
    ld_layout.addWidget(owner.fixed_use_bodipy_btn)

    owner.fixed_focus_btn = QPushButton("Preview Z focus")
    owner.fixed_focus_btn.setToolTip(
        "Compute a global Tenengrad focus preview. Final targeting still "
        "chooses the best plane independently for every stitched LD."
    )
    owner.fixed_focus_btn.clicked.connect(
        lambda _checked=False: _score_global_focus(owner)
    )
    ld_layout.addWidget(owner.fixed_focus_btn)

    owner.fixed_focus_status = QLabel("No BODIPY Z-stack loaded.")
    owner.fixed_focus_status.setWordWrap(True)
    ld_layout.addWidget(owner.fixed_focus_status)

    basic_form = QFormLayout()
    owner.fixed_z_mode = QComboBox()
    owner.fixed_z_mode.addItem("Cellpose 2D + stitch across Z", "stitch")
    owner.fixed_z_mode.addItem("Cellpose 2.5D / 3D flow", "3d")
    owner.fixed_z_mode.setToolTip(
        "2D+stitch segments each XY plane and joins overlapping masks across Z. "
        "2.5D runs Cellpose do_3D=True with axial anisotropy."
    )
    basic_form.addRow("Z segmentation mode", owner.fixed_z_mode)

    owner.fixed_ld_model_combo = QComboBox()
    for name in _cellpose_model_names():
        owner.fixed_ld_model_combo.addItem(name, name)
    basic_form.addRow("LD Cellpose model", owner.fixed_ld_model_combo)

    owner.fixed_ld_browse = QPushButton("Choose LD model weights…")
    owner.fixed_ld_browse.setToolTip(
        "Choose lipid-droplet-specific weights trained with this Cellpose version."
    )
    owner.fixed_ld_browse.clicked.connect(
        lambda _checked=False: owner.browse_cellpose_model(
            owner.fixed_ld_model_combo
        )
    )
    basic_form.addRow(owner.fixed_ld_browse)
    ld_layout.addLayout(basic_form)

    owner.fixed_advanced_box = make_collapsible(
        "Advanced segmentation parameters", expanded=False
    )
    advanced_form = QFormLayout(owner.fixed_advanced_box)

    owner.fixed_cellprob_input = QDoubleSpinBox()
    owner.fixed_cellprob_input.setRange(-10.0, 10.0)
    owner.fixed_cellprob_input.setSingleStep(0.25)
    owner.fixed_cellprob_input.setValue(1.0)
    owner.fixed_cellprob_input.setToolTip(
        "Higher is more conservative. Start at +1.0 and validate manually."
    )
    advanced_form.addRow("Cell-probability threshold", owner.fixed_cellprob_input)

    owner.fixed_flow_input = QDoubleSpinBox()
    owner.fixed_flow_input.setRange(0.0, 3.0)
    owner.fixed_flow_input.setSingleStep(0.05)
    owner.fixed_flow_input.setValue(0.4)
    owner.fixed_flow_input.setToolTip("Cellpose flow threshold; lower is stricter.")
    advanced_form.addRow("Flow threshold", owner.fixed_flow_input)

    owner.fixed_stitch_input = QDoubleSpinBox()
    owner.fixed_stitch_input.setRange(0.0, 1.0)
    owner.fixed_stitch_input.setSingleStep(0.05)
    owner.fixed_stitch_input.setValue(0.5)
    owner.fixed_stitch_input.setToolTip(
        "Adjacent-plane overlap threshold for 2D+stitch. Higher requires "
        "stronger Z continuity."
    )
    advanced_form.addRow("Z stitch threshold", owner.fixed_stitch_input)

    owner.fixed_z_step_input = QDoubleSpinBox()
    owner.fixed_z_step_input.setRange(0.01, 20.0)
    owner.fixed_z_step_input.setDecimals(3)
    owner.fixed_z_step_input.setValue(0.5)
    owner.fixed_z_step_input.setSuffix(" µm")
    owner.fixed_z_step_input.setToolTip(
        "Fallback only; the source MDA Z step is used automatically when present."
    )
    advanced_form.addRow("Fallback Z step", owner.fixed_z_step_input)

    owner.fixed_anisotropy_input = QDoubleSpinBox()
    owner.fixed_anisotropy_input.setRange(0.0, 100.0)
    owner.fixed_anisotropy_input.setDecimals(3)
    owner.fixed_anisotropy_input.setSpecialValueText("Auto = Z step / XY scale")
    owner.fixed_anisotropy_input.setValue(0.0)
    owner.fixed_anisotropy_input.setToolTip(
        "Leave 0 to calculate Cellpose anisotropy as Z-step / XY pixel size."
    )
    advanced_form.addRow("2.5D anisotropy override", owner.fixed_anisotropy_input)

    owner.fixed_z_status = QLabel("Z step not read yet.")
    advanced_form.addRow(owner.fixed_z_status)

    owner.fixed_focus_conf_input = QDoubleSpinBox()
    owner.fixed_focus_conf_input.setRange(0.0, 1.0)
    owner.fixed_focus_conf_input.setSingleStep(0.01)
    owner.fixed_focus_conf_input.setValue(0.05)
    owner.fixed_focus_conf_input.setToolTip(
        "Minimum separation between best and second-best local focus scores."
    )
    advanced_form.addRow("Minimum focus confidence", owner.fixed_focus_conf_input)

    owner.fixed_min_z_span_input = QSpinBox()
    owner.fixed_min_z_span_input.setRange(1, 1000)
    owner.fixed_min_z_span_input.setValue(1)
    owner.fixed_min_z_span_input.setToolTip(
        "Minimum number of Z planes occupied by the stitched/3D LD label."
    )
    advanced_form.addRow("Minimum LD Z span", owner.fixed_min_z_span_input)

    owner.fixed_max_z_span_input = QSpinBox()
    owner.fixed_max_z_span_input.setRange(0, 1000)
    owner.fixed_max_z_span_input.setValue(0)
    owner.fixed_max_z_span_input.setSpecialValueText("No limit")
    owner.fixed_max_z_span_input.setToolTip(
        "Optional maximum Z span to reject broad haze or merged clusters."
    )
    advanced_form.addRow("Maximum LD Z span", owner.fixed_max_z_span_input)

    owner.fixed_reject_edge_check = QCheckBox(
        "Reject LDs focused at first/last Z plane"
    )
    owner.fixed_reject_edge_check.setChecked(True)
    owner.fixed_reject_edge_check.setToolTip(
        "An edge maximum suggests the Z-stack did not bracket true focus."
    )
    advanced_form.addRow(owner.fixed_reject_edge_check)

    owner.fixed_grid_spacing_input = QDoubleSpinBox()
    owner.fixed_grid_spacing_input.setRange(0.1, 20.0)
    owner.fixed_grid_spacing_input.setValue(1.0)
    owner.fixed_grid_spacing_input.setSuffix(" µm")
    owner.fixed_grid_spacing_input.setToolTip(
        "Optional Raman spacing inside one LD; separate from the field grid."
    )
    advanced_form.addRow("Intra-LD grid spacing", owner.fixed_grid_spacing_input)

    owner.fixed_grid_margin_input = QDoubleSpinBox()
    owner.fixed_grid_margin_input.setRange(0.0, 20.0)
    owner.fixed_grid_margin_input.setValue(0.5)
    owner.fixed_grid_margin_input.setSuffix(" µm")
    owner.fixed_grid_margin_input.setToolTip(
        "Keep Raman grid points this far inside the focused LD boundary."
    )
    advanced_form.addRow("Intra-LD edge margin", owner.fixed_grid_margin_input)
    ld_layout.addWidget(owner.fixed_advanced_box)

    owner.fixed_segment_btn = QPushButton("Run focused LD segmentation")
    owner.fixed_segment_btn.setMinimumHeight(34)
    owner.fixed_segment_btn.setToolTip(
        "Run Cellpose 2D+stitch or 2.5D, apply physical diameter and Z-span "
        "filters, choose each LD's best-focus plane, and place a safe center. "
        "No hardware moves."
    )
    owner.fixed_segment_btn.clicked.connect(
        lambda _checked=False: _segment_droplets(owner)
    )
    ld_layout.addWidget(owner.fixed_segment_btn)
    segment_layout.addWidget(ld_group)
    layout.addWidget(segment_step)

    # ---------------- 3. REVIEW ----------------
    review_step = QGroupBox("3 — Review")
    review_step_layout = QVBoxLayout(review_step)

    cell_group = QGroupBox("CellMask Orange and LDs per cell")
    cell_layout = QVBoxLayout(cell_group)
    owner.fixed_use_cellmask_btn = QPushButton(
        "Use selected acquired layer as CellMask Orange"
    )
    owner.fixed_use_cellmask_btn.setToolTip(
        "Select the matching CellMask Orange layer. A maximum projection is "
        "segmented as the parent-cell mask."
    )
    owner.fixed_use_cellmask_btn.clicked.connect(
        lambda _checked=False: _use_cellmask(owner)
    )
    cell_layout.addWidget(owner.fixed_use_cellmask_btn)

    cell_form = QFormLayout()
    owner.fixed_cell_model_combo = QComboBox()
    for name in _cellpose_model_names():
        owner.fixed_cell_model_combo.addItem(name, name)
    cell_form.addRow("Whole-cell Cellpose model", owner.fixed_cell_model_combo)

    owner.fixed_cell_browse = QPushButton("Choose whole-cell model weights…")
    owner.fixed_cell_browse.clicked.connect(
        lambda _checked=False: owner.browse_cellpose_model(
            owner.fixed_cell_model_combo
        )
    )
    cell_form.addRow(owner.fixed_cell_browse)

    owner.fixed_cell_diam_input = QDoubleSpinBox()
    owner.fixed_cell_diam_input.setRange(1.0, 500.0)
    owner.fixed_cell_diam_input.setValue(35.0)
    owner.fixed_cell_diam_input.setSuffix(" µm")
    owner.fixed_cell_diam_input.setToolTip(
        "Expected whole-cell diameter, converted to pixels using the active "
        "objective calibration."
    )
    cell_form.addRow("Expected cell diameter", owner.fixed_cell_diam_input)

    owner.fixed_cell_prob_input = QDoubleSpinBox()
    owner.fixed_cell_prob_input.setRange(-10.0, 10.0)
    owner.fixed_cell_prob_input.setValue(0.5)
    cell_form.addRow("Whole-cell probability threshold", owner.fixed_cell_prob_input)

    owner.fixed_cell_flow_input = QDoubleSpinBox()
    owner.fixed_cell_flow_input.setRange(0.0, 3.0)
    owner.fixed_cell_flow_input.setValue(0.4)
    cell_form.addRow("Whole-cell flow threshold", owner.fixed_cell_flow_input)

    owner.fixed_require_cell_check = QCheckBox(
        "Require every LD target to belong to a CellMask cell"
    )
    owner.fixed_require_cell_check.setChecked(True)
    owner.fixed_require_cell_check.toggled.connect(
        lambda _checked=False: _assign_cells(owner)
    )
    cell_form.addRow(owner.fixed_require_cell_check)

    owner.fixed_max_per_cell_input = QSpinBox()
    owner.fixed_max_per_cell_input.setRange(0, 1000)
    owner.fixed_max_per_cell_input.setSpecialValueText("All valid LDs")
    owner.fixed_max_per_cell_input.setValue(0)
    owner.fixed_max_per_cell_input.setToolTip(
        "0 keeps all valid focused LDs per cell. A positive value retains the "
        "highest-confidence targets in each cell."
    )
    owner.fixed_max_per_cell_input.valueChanged.connect(
        lambda _value=0: _assign_cells(owner)
    )
    cell_form.addRow("Maximum LDs per cell", owner.fixed_max_per_cell_input)
    cell_layout.addLayout(cell_form)

    owner.fixed_segment_cells_btn = QPushButton(
        "Segment cells and assign focused LDs"
    )
    owner.fixed_segment_cells_btn.setToolTip(
        "Segment CellMask Orange, assign each LD center a parent cell_id, and "
        "apply the per-cell target limit."
    )
    owner.fixed_segment_cells_btn.clicked.connect(
        lambda _checked=False: _segment_cells(owner)
    )
    cell_layout.addWidget(owner.fixed_segment_cells_btn)
    review_step_layout.addWidget(cell_group)

    targets_group = QGroupBox("Focused LD targets")
    targets_layout = QVBoxLayout(targets_group)
    owner.fixed_intra_grid_check = QCheckBox(
        "Use intra-LD Raman grid instead of one centered point"
    )
    owner.fixed_intra_grid_check.setChecked(False)
    owner.fixed_intra_grid_check.setToolTip(
        "Leave OFF for the first spatial-validation pilot. When enabled, the "
        "stage centers each LD and existing Raman batch/galvo logic samples "
        "the retained interior grid."
    )
    owner.fixed_intra_grid_check.toggled.connect(
        lambda _checked=False: _refresh_targets(owner)
    )
    targets_layout.addWidget(owner.fixed_intra_grid_check)

    owner.fixed_target_summary = QLabel("No focused LD targets yet.")
    owner.fixed_target_summary.setWordWrap(True)
    targets_layout.addWidget(owner.fixed_target_summary)

    owner.fixed_target_table = QTableWidget(0, 7)
    owner.fixed_target_table.setHorizontalHeaderLabels(
        ["LD", "Cell", "Best Z", "Diameter µm", "Focus conf.", "Z span", "Selected"]
    )
    owner.fixed_target_table.setMaximumHeight(220)
    targets_layout.addWidget(owner.fixed_target_table)

    review_buttons = QHBoxLayout()
    owner.fixed_reject_btn = QPushButton("Reject selected target(s)")
    owner.fixed_reject_btn.setToolTip(
        "Select red LD-center points in Napari, then reject them from targeting."
    )
    owner.fixed_reject_btn.clicked.connect(
        lambda _checked=False: _reject_selected(owner)
    )
    owner.fixed_reset_btn = QPushButton("Reset rejections")
    owner.fixed_reset_btn.clicked.connect(
        lambda _checked=False: (
            owner._fixed.update(rejected=set()), _refresh_targets(owner)
        )
    )
    review_buttons.addWidget(owner.fixed_reject_btn)
    review_buttons.addWidget(owner.fixed_reset_btn)
    targets_layout.addLayout(review_buttons)
    review_step_layout.addWidget(targets_group)
    layout.addWidget(review_step)

    # ---------------- 4. TARGET ----------------
    target_step = QGroupBox("4 — Target")
    target_layout = QVBoxLayout(target_step)
    note = QLabel(
        "Validation performs calculations only. No stage, galvo, shutter or "
        "laser movement. Hardware execution remains exclusively in shared MDA Run."
    )
    note.setWordWrap(True)
    target_layout.addWidget(note)

    owner.fixed_validate_btn = QPushButton(
        "Validate pixel → stage targets (no motion)"
    )
    owner.fixed_validate_btn.setMinimumHeight(34)
    owner.fixed_validate_btn.setToolTip(
        "Calculate objective-calibrated stage positions and verify finite XYZ "
        "and source-image provenance. No stage, galvo, shutter or laser command."
    )
    owner.fixed_validate_btn.clicked.connect(
        lambda _checked=False: _validate_stage(owner)
    )
    target_layout.addWidget(owner.fixed_validate_btn)

    owner.fixed_stage_status = QLabel("Stage targets not validated.")
    owner.fixed_stage_status.setWordWrap(True)
    target_layout.addWidget(owner.fixed_stage_status)

    owner.fixed_prepare_btn = QPushButton("Send targets to shared Raman MDA")
    owner.fixed_prepare_btn.setMinimumHeight(34)
    owner.fixed_prepare_btn.setToolTip(
        "Attach validated LD stage positions to the unchanged shared MDA. "
        "This does not press Run or move hardware."
    )
    owner.fixed_prepare_btn.clicked.connect(
        lambda _checked=False: _prepare_mda(owner)
    )
    target_layout.addWidget(owner.fixed_prepare_btn)
    layout.addWidget(target_step)

    owner.fixed_status = QLabel("Fixed-cell validation mode ready.")
    owner.fixed_status.setWordWrap(True)
    layout.addWidget(owner.fixed_status)

    owner._fixed = dict(
        stack=None,
        source_layer=None,
        source_context={},
        source_sequence=None,
        pixel_scale=None,
        targets=[],
        rejected=set(),
        cellmask_image=None,
        cell_labels=None,
        stage_positions=[],
        mda_prepared=False,
    )

    def _set_interface_mode(text):
        owner.fixed_advanced_box.setVisible(text == "Expert")

    owner.fixed_interface_mode.currentTextChanged.connect(_set_interface_mode)
    _set_interface_mode(owner.fixed_interface_mode.currentText())
    _update_physical_summary(owner)
    _update_fixed_ui_state(owner)
    return box
