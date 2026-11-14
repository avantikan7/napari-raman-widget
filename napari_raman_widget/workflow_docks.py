"""Task-focused dock layout for the LipidDrop application.

This module changes presentation only. It never replaces or invokes microscope
hardware actions. Jeff's existing shared MDA remains the sole owner of
acquisition setup, saving, autofocus, Run, and Cancel.
"""
from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDockWidget,
    QGroupBox,
    QLabel,
    QMainWindow,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


def _status_mark(done: bool) -> str:
    return "✓" if done else "○"


def _qt_main_window(owner) -> QMainWindow | None:
    """Return Napari's Qt main window without creating or moving hardware."""
    qt_window = getattr(getattr(owner.viewer, "window", None), "_qt_window", None)
    return qt_window if isinstance(qt_window, QMainWindow) else None


def _dock_main_window(dock) -> QMainWindow | None:
    parent = dock
    while parent is not None:
        if isinstance(parent, QMainWindow):
            return parent
        parent = parent.parentWidget() if hasattr(parent, "parentWidget") else None
    return None


def refresh_control_hub(owner) -> None:
    """Refresh the compact, hardware-free workflow state summary."""
    label = getattr(owner, "lipiddrop_hub_status", None)
    if label is None:
        return

    connected = owner.core is not None
    objective = getattr(owner, "objective_combo", None)
    objective_text = objective.currentText() if objective is not None else "unknown"
    calibration = owner.vandermonde is not None

    fixed = getattr(owner, "_fixed", {}) or {}
    rejected = set(fixed.get("rejected", set()))
    fixed_targets = sum(
        1
        for row in fixed.get("targets", [])
        if row.get("selected", True) and row.get("id") not in rejected
    )
    fixed_prepared = bool(fixed.get("mda_prepared"))

    live = getattr(owner, "_dry", {}) or {}
    live_baseline = len(live.get("baseline") or [])
    live_prepared = bool(live.get("mda_prepared"))

    label.setText(
        f"{_status_mark(connected)} Microscope: "
        f"{'connected' if connected else 'disconnected'}\n"
        f"{_status_mark(objective_text not in {'(connect first)', '(index unavailable)', ''})} "
        f"Objective: {objective_text}\n"
        f"{_status_mark(calibration)} Pixel→stage calibration: "
        f"{'valid' if calibration else 'not loaded'}\n"
        f"{_status_mark(fixed_targets > 0)} 3D LD targets: {fixed_targets} | "
        f"MDA prepared: {'yes' if fixed_prepared else 'no'}\n"
        f"{_status_mark(live_baseline > 0)} LD tracking T0 identities: "
        f"{live_baseline if live_baseline else 'not set'} | "
        f"MDA prepared: {'yes' if live_prepared else 'no'}"
    )


def _lock_workflow_dock(dock: QDockWidget) -> None:
    """Keep a workflow dock tabbed; allow close but prevent drag/floating."""
    features = dock.features()
    features &= ~QDockWidget.DockWidgetFloatable
    features &= ~QDockWidget.DockWidgetMovable
    dock.setFeatures(features)
    dock.setFloating(False)


def tabify_workflow_docks(
    owner,
    anchor_dock: QDockWidget | None = None,
    main_window: QMainWindow | None = None,
) -> bool:
    """Place Raman, 3D LD Targeting, and LD Tracking in one fixed tab group."""
    fixed = getattr(owner, "_fixed_workflow_dock", None)
    live = getattr(owner, "_live_workflow_dock", None)
    if fixed is None or live is None:
        return False

    if anchor_dock is None:
        anchor_dock = getattr(owner, "_workflow_anchor_dock", None)
    if main_window is None:
        main_window = _qt_main_window(owner)
    if main_window is None and anchor_dock is not None:
        main_window = _dock_main_window(anchor_dock)
    if main_window is None or anchor_dock is None:
        return False

    for dock in (fixed, live):
        if _dock_main_window(dock) is not main_window:
            main_window.addDockWidget(Qt.RightDockWidgetArea, dock)
        _lock_workflow_dock(dock)
        dock.show()

    main_window.tabifyDockWidget(anchor_dock, fixed)
    main_window.tabifyDockWidget(fixed, live)
    main_window.setTabPosition(Qt.RightDockWidgetArea, QTabWidget.North)
    owner._workflow_anchor_dock = anchor_dock
    anchor_dock.show()
    anchor_dock.raise_()
    return True


def tabify_shared_mda(owner) -> bool:
    """Add Jeff's unchanged MDA dock to the same fixed right-side tab group."""
    if owner.main_window is None:
        return False
    mda_dock = owner.main_window._dock_widgets.get("MDA")
    if mda_dock is None:
        return False

    main_window = _qt_main_window(owner) or _dock_main_window(mda_dock)
    anchor = getattr(owner, "_workflow_anchor_dock", None)
    if main_window is None or anchor is None:
        return False

    # Do not change MDA controls or feature flags; only place its existing dock
    # into the same tab group used by the biological workflows.
    mda_dock.setFloating(False)
    live = getattr(owner, "_live_workflow_dock", anchor)
    main_window.tabifyDockWidget(live, mda_dock)
    main_window.setTabPosition(Qt.RightDockWidgetArea, QTabWidget.North)
    mda_dock.show()
    anchor.raise_()
    return True


def _show_dock(owner, attribute: str) -> None:
    dock = getattr(owner, attribute, None)
    if dock is None:
        owner.status.setText("Status: workflow tab is not available")
        return
    tabify_workflow_docks(owner)
    dock.show()
    dock.raise_()
    refresh_control_hub(owner)


def show_shared_mda(owner) -> None:
    """Reveal Jeff's existing MDA tab without changing its controls."""
    if owner.main_window is None:
        owner.status.setText("Status: connect the microscope before opening MDA")
        return
    dock = owner.main_window._dock_widgets.get("MDA")
    if dock is None:
        owner.status.setText("Status: shared MDA dock is unavailable")
        return
    tabify_shared_mda(owner)
    dock.show()
    dock.raise_()
    owner.status.setText("Status: opened Jeff's shared Raman MDA")
    refresh_control_hub(owner)


def build_control_hub(owner) -> QGroupBox:
    """Build a compact status panel; navigation uses the real dock tabs."""
    box = QGroupBox("LipidDrop status")
    layout = QVBoxLayout(box)

    note = QLabel(
        "Use the dock tabs above: Raman | 3D LD Targeting | LD Tracking | MDA. "
        "Workflows stay docked in one pane; no dragging is required."
    )
    note.setWordWrap(True)
    note.setToolTip(
        "3D LD Targeting and LD Tracking are biological-analysis workspaces. "
        "MDA remains Jeff's unchanged acquisition workspace. Switching tabs "
        "never moves the stage, galvo, objective, shutter, filter, or laser."
    )
    layout.addWidget(note)

    owner.lipiddrop_hub_status = QLabel()
    owner.lipiddrop_hub_status.setWordWrap(True)
    owner.lipiddrop_hub_status.setStyleSheet(
        "QLabel { border: 1px solid palette(mid); padding: 6px; "
        "font-family: monospace; }"
    )
    owner.lipiddrop_hub_status.setToolTip(
        "Read-only in-memory workflow state. It does not query or command "
        "microscope hardware."
    )
    layout.addWidget(owner.lipiddrop_hub_status)

    refresh_control_hub(owner)
    return box


def _workflow_scroll(widget: QWidget) -> QScrollArea:
    widget.setCheckable(False)
    widget.setChecked(True)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setWidget(widget)
    scroll.setMinimumWidth(480)
    return scroll


def install_workflow_docks(owner, fixed_widget: QWidget, live_widget: QWidget) -> None:
    """Create docked fixed-cell and live tabs; never create floating windows."""
    fixed_scroll = _workflow_scroll(fixed_widget)
    live_scroll = _workflow_scroll(live_widget)

    add_dock = getattr(getattr(owner.viewer, "window", None), "add_dock_widget", None)
    real_docks = callable(add_dock)
    if real_docks:
        owner._fixed_workflow_dock = add_dock(
            fixed_scroll,
            name="3D LD Targeting",
            area="right",
            allowed_areas=("right",),
            add_vertical_stretch=False,
        )
        owner._live_workflow_dock = add_dock(
            live_scroll,
            name="LD Tracking",
            area="right",
            allowed_areas=("right",),
            add_vertical_stretch=False,
        )
    else:
        # Hardware-free tests use ordinary QDockWidgets and tabify them later.
        owner._fixed_workflow_dock = QDockWidget("3D LD Targeting")
        owner._fixed_workflow_dock.setWidget(fixed_scroll)
        owner._live_workflow_dock = QDockWidget("LD Tracking")
        owner._live_workflow_dock.setWidget(live_scroll)

    for dock in (owner._fixed_workflow_dock, owner._live_workflow_dock):
        _lock_workflow_dock(dock)
        if real_docks:
            dock.show()
        else:
            dock.hide()

    owner._fixed_workflow_scroll = fixed_scroll
    owner._live_workflow_scroll = live_scroll
    refresh_control_hub(owner)
