"""Task-focused Napari docks for the LipidDrop application.

This module only reorganizes Qt widgets. It does not replace, wrap, or call any
microscope hardware action. Jeff's existing shared MDA remains the only place
that owns acquisition setup and Run/Cancel.
"""
from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDockWidget,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QTabBar,
    QVBoxLayout,
    QWidget,
)


def _status_mark(done: bool) -> str:
    return "✓" if done else "○"


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
    fixed_targets = sum(
        1
        for row in fixed.get("targets", [])
        if row.get("selected", True)
        and row.get("id") not in set(fixed.get("rejected", set()))
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
        f"{_status_mark(fixed_targets > 0)} Fixed-cell targets: {fixed_targets} | "
        f"MDA prepared: {'yes' if fixed_prepared else 'no'}\n"
        f"{_status_mark(live_baseline > 0)} Live T0 identities: "
        f"{live_baseline if live_baseline else 'not set'} | "
        f"MDA prepared: {'yes' if live_prepared else 'no'}"
    )


def _show_dock(owner, attribute: str) -> None:
    dock = getattr(owner, attribute, None)
    if dock is None:
        owner.status.setText("Status: workflow dock is not available")
        return
    dock.show()
    dock.raise_()
    dock.activateWindow()
    refresh_control_hub(owner)


def show_shared_mda(owner) -> None:
    """Reveal Jeff's existing MDA dock without changing its controls."""
    if owner.main_window is None:
        owner.status.setText("Status: connect the microscope before opening MDA")
        return
    dock = owner.main_window._dock_widgets.get("MDA")
    if dock is None:
        owner.status.setText("Status: shared MDA dock is unavailable")
        return
    dock.show()
    dock.raise_()
    dock.activateWindow()
    owner.status.setText("Status: opened Jeff's shared Raman MDA")
    refresh_control_hub(owner)


def build_control_hub(owner) -> QGroupBox:
    """Build a compact tab-style launcher for the three shared workspaces."""
    box = QGroupBox("LipidDrop Control")
    layout = QVBoxLayout(box)

    note = QLabel(
        "Choose a workspace. Jeff's shared MDA remains the only place that owns "
        "acquisition setup, saving, Autofocus, Run and Cancel."
    )
    note.setWordWrap(True)
    note.setToolTip(
        "These tabs only show or focus existing Napari docks. They do not move "
        "the stage, galvo, objective, shutter, filter or laser."
    )
    layout.addWidget(note)

    owner.workflow_tabs = QTabBar()
    owner.workflow_tabs.setDocumentMode(True)
    owner.workflow_tabs.setExpanding(True)
    owner.workflow_tabs.setUsesScrollButtons(False)
    owner.workflow_tabs.setMinimumHeight(34)
    owner.workflow_tabs.setAccessibleName("LipidDrop workspace tabs")
    owner.workflow_tabs.addTab("Fixed-cell")
    owner.workflow_tabs.addTab("Live lipolysis")
    owner.workflow_tabs.addTab("Raman MDA")
    owner.workflow_tabs.setTabToolTip(
        0,
        "Open the fixed-cell Z-stack workflow: Acquire → Segment → Review → Target. "
        "No hardware moves until a reviewed plan is run from shared MDA.",
    )
    owner.workflow_tabs.setTabToolTip(
        1,
        "Open the experimental live workflow: Baseline → Track → Analyze → Revisit.",
    )
    owner.workflow_tabs.setTabToolTip(
        2,
        "Open Jeff's unchanged shared MDA with Time, Positions, Grid, Z Stack, "
        "Channels, Raman selection, Autofocus, Save, Run and Cancel.",
    )

    def open_workspace(index: int) -> None:
        if index == 0:
            _show_dock(owner, "_fixed_workflow_dock")
        elif index == 1:
            _show_dock(owner, "_live_workflow_dock")
        elif index == 2:
            show_shared_mda(owner)

    owner.workflow_tabs.tabBarClicked.connect(open_workspace)
    owner.workflow_tabs.setCurrentIndex(-1)
    layout.addWidget(owner.workflow_tabs)

    hint = QLabel("Click a tab to open or refocus that workspace.")
    hint.setWordWrap(True)
    hint.setStyleSheet("QLabel { color: palette(mid); padding-bottom: 2px; }")
    layout.addWidget(hint)

    owner.lipiddrop_hub_status = QLabel()
    owner.lipiddrop_hub_status.setWordWrap(True)
    owner.lipiddrop_hub_status.setStyleSheet(
        "QLabel { border: 1px solid palette(mid); padding: 6px; "
        "font-family: monospace; }"
    )
    owner.lipiddrop_hub_status.setToolTip(
        "Read-only state summary. It reads only in-memory software state and "
        "does not query or command microscope hardware."
    )
    layout.addWidget(owner.lipiddrop_hub_status)

    # State is refreshed by connect/disconnect and by the fixed/live workflow
    # state-update functions. Avoid a polling timer so no Qt callback survives
    # widget teardown in long microscope sessions or test runs.
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
    """Install independent fixed-cell and live workflow docks in one viewer."""
    fixed_scroll = _workflow_scroll(fixed_widget)
    live_scroll = _workflow_scroll(live_widget)

    add_dock = getattr(getattr(owner.viewer, "window", None), "add_dock_widget", None)
    if callable(add_dock):
        owner._fixed_workflow_dock = add_dock(
            fixed_scroll,
            name="Fixed-cell Targeting",
            area="right",
            allowed_areas=("left", "right"),
            add_vertical_stretch=False,
        )
        owner._live_workflow_dock = add_dock(
            live_scroll,
            name="Live Lipolysis Tracker",
            area="right",
            allowed_areas=("left", "right"),
            add_vertical_stretch=False,
        )
    else:
        # Lightweight fallback for hardware-free unit tests that construct the
        # widget with a minimal viewer stub rather than a real Napari Window.
        owner._fixed_workflow_dock = QDockWidget("Fixed-cell Targeting")
        owner._fixed_workflow_dock.setWidget(fixed_scroll)
        owner._live_workflow_dock = QDockWidget("Live Lipolysis Tracker")
        owner._live_workflow_dock.setWidget(live_scroll)

    for dock in (owner._fixed_workflow_dock, owner._live_workflow_dock):
        dock.setFloating(True)
        dock.resize(560, 820)
        dock.hide()

    owner._fixed_workflow_scroll = fixed_scroll
    owner._live_workflow_scroll = live_scroll
    refresh_control_hub(owner)
