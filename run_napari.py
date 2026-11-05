# run_napari.py
"""Entry point: launches napari with the CNS Raman widget docked on the right."""

from napari_raman_widget.logging_config import configure_pymmcore_process_log

# This must run before importing napari or the hardware widget: either may
# eventually import pymmcore-plus, whose rotating log path is fixed at import.
configure_pymmcore_process_log()

import napari

from napari_raman_widget import HardwareWidget


if __name__ == "__main__":
    viewer = napari.Viewer()
    viewer.axes.visible = False  # napari-micromanager axes crash workaround
    from pathlib import Path
    from qtpy.QtGui import QIcon
    from qtpy.QtWidgets import QApplication

    icon = QIcon(str(Path(__file__).resolve().parents[1] / "assets" / "lipid-droplet-icon.png"))
    QApplication.instance().setWindowIcon(icon)  # Dock icon on macOS
    viewer.window._qt_window.setWindowIcon(icon)
    widget = HardwareWidget(viewer)
    viewer.window.add_dock_widget(widget, name="Raman", area="right")
    napari.run()
