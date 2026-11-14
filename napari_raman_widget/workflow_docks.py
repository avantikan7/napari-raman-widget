"""Task-focused dock layout for the LipidDrop application.

This module changes presentation only. It never replaces or invokes microscope
hardware actions. Jeff's existing shared MDA remains the sole owner of
acquisition setup, saving, autofocus, Run, and Cancel.
"""
from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDockWidget,
    QFrame,
    QGridLayout,
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


_STATUS_COLORS = {
    "ready": "#63D69A",
    "attention": "#F2B85B",
    "inactive": "#9AA3AF",
    "error": "#F07878",
}


def _set_status_row(
    owner,
    key: str,
    value: str,
    level: str,
    *,
    badge: str | None = None,
    badge_level: str = "inactive",
) -> bool:
    """Update one compact readiness row without touching microscope hardware."""
    rows = getattr(owner, "lipiddrop_status_rows", None)
    if not rows or key not in rows:
        return False

    row = rows[key]
    color = _STATUS_COLORS.get(level, _STATUS_COLORS["inactive"])
    row["dot"].setText("●")
    row["dot"].setStyleSheet(f"QLabel {{ color: {color}; font-size: 13px; }}")
    row["value"].setText(value)
    row["value"].setStyleSheet(f"QLabel {{ color: {color}; font-weight: 600; }}")

    badge_widget = row.get("badge")
    if badge_widget is not None:
        if badge is None:
            badge_widget.hide()
        else:
            badge_color = _STATUS_COLORS.get(
                badge_level, _STATUS_COLORS["inactive"]
            )
            badge_widget.setText(badge)
            badge_widget.setStyleSheet(
                "QLabel {"
                f" color: {badge_color};"
                f" border: 1px solid {badge_color};"
                " border-radius: 7px;"
                " padding: 1px 6px;"
                " font-weight: 600;"
                "}"
            )
            badge_widget.show()
    return True


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
    """Refresh the compact, hardware-free workflow readiness card."""
    connected = owner.core is not None
    objective = getattr(owner, "objective_combo", None)
    objective_text = objective.currentText() if objective is not None else ""
    objective_ready = bool(
        connected
        and objective_text
        and objective_text not in {"(connect first)", "(index unavailable)"}
    )
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

    if getattr(owner, "lipiddrop_status_rows", None):
        _set_status_row(
            owner,
            "microscope",
            "Connected" if connected else "Offline",
            "ready" if connected else "inactive",
        )
        _set_status_row(
            owner,
            "objective",
            objective_text if objective_ready else "Connect microscope",
            "ready" if objective_ready else "inactive",
        )
        _set_status_row(
            owner,
            "calibration",
            "Loaded" if calibration else "Not loaded",
            "ready" if calibration else ("attention" if connected else "inactive"),
        )
        _set_status_row(
            owner,
            "targets",
            f"{fixed_targets} selected",
            "ready" if fixed_targets else "inactive",
            badge="MDA ready" if fixed_prepared else (
                "MDA pending" if fixed_targets else "MDA not prepared"
            ),
            badge_level="ready" if fixed_prepared else (
                "attention" if fixed_targets else "inactive"
            ),
        )
        _set_status_row(
            owner,
            "tracking",
            f"T0: {live_baseline} identities" if live_baseline else "T0 not set",
            "ready" if live_baseline else "inactive",
            badge="MDA ready" if live_prepared else (
                "MDA pending" if live_baseline else "MDA not prepared"
            ),
            badge_level="ready" if live_prepared else (
                "attention" if live_baseline else "inactive"
            ),
        )

    # Compatibility text for tests or minimal viewer stubs that do not build rows.
    label = getattr(owner, "lipiddrop_hub_status", None)
    if label is not None:
        label.setText(
            f"Microscope: {'Connected' if connected else 'Offline'}\n"
            f"Objective: {objective_text if objective_ready else 'Connect microscope'}\n"
            f"Pixel→stage: {'Loaded' if calibration else 'Not loaded'}\n"
            f"3D LD targeting: {fixed_targets} selected | "
            f"MDA: {'ready' if fixed_prepared else 'not prepared'}\n"
            f"LD tracking: "
            f"{'T0 ' + str(live_baseline) if live_baseline else 'T0 not set'} | "
            f"MDA: {'ready' if live_prepared else 'not prepared'}"
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

    card = QFrame()
    card.setObjectName("lipiddropReadinessCard")
    card.setFrameShape(QFrame.StyledPanel)
    card.setStyleSheet(
        "QFrame#lipiddropReadinessCard {"
        " border: 1px solid palette(mid);"
        " border-radius: 5px;"
        " padding: 4px;"
        "}"
    )
    grid = QGridLayout(card)
    grid.setContentsMargins(8, 7, 8, 7)
    grid.setHorizontalSpacing(8)
    grid.setVerticalSpacing(5)
    grid.setColumnStretch(1, 1)

    owner.lipiddrop_status_rows = {}

    def add_row(row_index, key, title, tooltip, with_badge=False):
        dot = QLabel("●")
        dot.setFixedWidth(14)
        title_label = QLabel(title)
        title_label.setStyleSheet("QLabel { font-weight: 600; }")
        value = QLabel()
        value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        badge = QLabel() if with_badge else None
        if badge is not None:
            badge.setAlignment(Qt.AlignCenter)
        for widget in (dot, title_label, value, badge):
            if widget is not None:
                widget.setToolTip(tooltip)
        grid.addWidget(dot, row_index, 0)
        grid.addWidget(title_label, row_index, 1)
        grid.addWidget(value, row_index, 2)
        if badge is not None:
            grid.addWidget(badge, row_index, 3)
        owner.lipiddrop_status_rows[key] = {
            "dot": dot,
            "title": title_label,
            "value": value,
            "badge": badge,
        }

    add_row(0, "microscope", "Microscope", "Connection state only; this indicator never connects or moves hardware.")
    add_row(1, "objective", "Objective", "Active objective reported by the connected IX83 configuration.")
    add_row(2, "calibration", "Pixel→stage", "Whether a current-objective Vandermonde calibration is loaded.")
    add_row(3, "targets", "3D LD targeting", "Number of reviewed 3D LD targets and whether they have been handed to MDA.", True)
    add_row(4, "tracking", "LD tracking", "Whether T0 identities exist and whether revisit targets have been handed to MDA.", True)
    layout.addWidget(card)

    # Retain a hidden text representation for minimal test stubs and accessibility.
    owner.lipiddrop_hub_status = QLabel()
    owner.lipiddrop_hub_status.hide()
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
