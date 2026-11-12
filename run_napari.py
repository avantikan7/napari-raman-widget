# run_napari.py
"""Entry point: launches napari with the CNS Raman widget docked on the right."""

import sys

# Windows groups taskbar buttons by AppUserModelID. Without an explicit ID,
# this process is grouped as generic Python/napari even when Qt has a custom icon.
if sys.platform == "win32":
    import ctypes

    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
        "avantikan7.LipidDropImaging"
    )

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

    assets = Path(__file__).resolve().parents[1] / "assets"
    icon_path = (
        assets / "lipid-droplet-icon.ico"
        if sys.platform == "win32"
        else assets / "lipid-droplet-icon.png"
    )
    icon = QIcon(str(icon_path))

    app = QApplication.instance()
    app.setApplicationName("LipidDrop Imaging")
    app.setApplicationDisplayName("LipidDrop Imaging")
    app.setWindowIcon(icon)

    viewer.window._qt_window.setWindowTitle("LipidDrop Imaging")
    viewer.window._qt_window.setWindowIcon(icon)

    widget = HardwareWidget(viewer)
    viewer.window.add_dock_widget(widget, name="Raman", area="right")
    napari.run()
