from types import SimpleNamespace

import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QDockWidget,
    QGroupBox,
    QMainWindow,
    QPushButton,
)


@pytest.fixture(scope="session")
def qt_app():
    return QApplication.instance() or QApplication([])


def test_lipid_workflows_are_real_dock_tabs_and_hardware_controls_remain(qt_app):
    """Workflow tabs must not replace or modify Jeff's hardware controls."""
    from napari_raman_widget.widget import HardwareWidget
    from napari_raman_widget.workflow_docks import (
        tabify_shared_mda,
        tabify_workflow_docks,
    )

    viewer = SimpleNamespace(window=SimpleNamespace())
    widget = HardwareWidget(viewer)
    main = QMainWindow()
    raman = QDockWidget("Raman")
    raman.setWidget(QGroupBox("hardware placeholder"))
    main.addDockWidget(Qt.RightDockWidgetArea, raman)

    try:
        assert widget.lipiddrop_control_box.title() == "LipidDrop status"
        assert widget._fixed_workflow_dock.windowTitle() == "3D LD Targeting"
        assert widget._live_workflow_dock.windowTitle() == "LD Tracking"

        # Existing hardware actions are still the original HardwareWidget controls.
        for name in (
            "connect_btn",
            "disconnect_btn",
            "click_center_btn",
            "click_laser_btn",
            "center_laser_btn",
            "laser_autofocus_btn",
            "open_shutter_btn",
            "close_shutter_btn",
        ):
            assert isinstance(getattr(widget, name), QPushButton)

        fixed_titles = {
            group.title() for group in widget.fixed_cell_box.findChildren(QGroupBox)
        }
        assert {"1 — Acquire", "2 — Segment", "3 — Review", "4 — Target"} <= fixed_titles

        live_titles = {
            group.title() for group in widget.dry_run_box.findChildren(QGroupBox)
        }
        assert {
            "1 — Baseline",
            "2 — Track",
            "3 — Analyze response",
            "4 — Revisit",
        } <= live_titles

        assert tabify_workflow_docks(widget, raman, main)
        tabified = set(main.tabifiedDockWidgets(raman))
        assert widget._fixed_workflow_dock in tabified
        assert widget._live_workflow_dock in tabified
        assert not widget._fixed_workflow_dock.isFloating()
        assert not widget._live_workflow_dock.isFloating()

        for dock in (widget._fixed_workflow_dock, widget._live_workflow_dock):
            assert not (dock.features() & QDockWidget.DockWidgetFloatable)
            assert not (dock.features() & QDockWidget.DockWidgetMovable)
            assert not dock.isHidden()

        # Jeff's existing MDA dock is only tabified; its controls/features stay intact.
        mda = QDockWidget("MDA")
        mda.setWidget(QGroupBox("Jeff MDA placeholder"))
        main.addDockWidget(Qt.RightDockWidgetArea, mda)
        mda_features = mda.features()
        widget.main_window = SimpleNamespace(_dock_widgets={"MDA": mda})
        widget.viewer.window = SimpleNamespace(_qt_window=main)
        assert tabify_shared_mda(widget)
        assert mda in set(main.tabifiedDockWidgets(raman))
        assert mda.features() == mda_features
        assert not mda.isFloating()
    finally:
        main.close()
        widget.close()
        qt_app.processEvents()
