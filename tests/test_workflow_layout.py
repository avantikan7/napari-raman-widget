from types import SimpleNamespace

import pytest
from qtpy.QtWidgets import QApplication, QGroupBox, QPushButton, QTabBar


@pytest.fixture(scope="session")
def qt_app():
    return QApplication.instance() or QApplication([])


def test_lipid_workflows_are_separate_docks_and_hardware_controls_remain(qt_app):
    """The UI refactor must not delete or replace Jeff's hardware controls."""
    from napari_raman_widget.widget import HardwareWidget

    viewer = SimpleNamespace(window=SimpleNamespace())
    widget = HardwareWidget(viewer)
    try:
        assert widget.lipiddrop_control_box.title() == "LipidDrop Control"
        assert widget._fixed_workflow_dock.windowTitle() == "Fixed-cell Targeting"
        assert widget._live_workflow_dock.windowTitle() == "Live Lipolysis Tracker"
        assert widget._fixed_workflow_dock.isHidden()
        assert widget._live_workflow_dock.isHidden()

        # Existing hardware actions are still the same controls on HardwareWidget.
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

        assert isinstance(widget.workflow_tabs, QTabBar)
        assert [
            widget.workflow_tabs.tabText(i)
            for i in range(widget.workflow_tabs.count())
        ] == ["Fixed-cell", "Live lipolysis", "Raman MDA"]

        widget.workflow_tabs.tabBarClicked.emit(0)
        assert not widget._fixed_workflow_dock.isHidden()
        assert widget._live_workflow_dock.isHidden()

        widget.workflow_tabs.tabBarClicked.emit(1)
        assert not widget._live_workflow_dock.isHidden()
    finally:
        widget._fixed_workflow_dock.close()
        widget._live_workflow_dock.close()
        widget.close()
        qt_app.processEvents()
