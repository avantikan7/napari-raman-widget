# run_napari.py
"""Entry point: launches napari with the CNS Raman widget docked on the right."""

import os
import sys
from pathlib import Path

# Make the LipidDrop workspace components import from this checked-out workspace
# even when the app is launched directly with pythonw.exe (without a .bat file).
WORKSPACE = Path(__file__).resolve().parents[1]
for component in (
    WORKSPACE / "napari-raman-widget",
    WORKSPACE / "raman-control",
    WORKSPACE / "raman-mda-engine",
    WORKSPACE / "cns-control" / "src",
):
    path = str(component)
    if path not in sys.path:
        sys.path.insert(0, path)

APP_ID = "avantikan7.LipidDropImaging.Adipocyte"

if sys.platform == "win32":
    import ctypes

    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)

from napari_raman_widget.logging_config import configure_pymmcore_process_log

# This must run before importing napari or the hardware widget: either may
# eventually import pymmcore-plus, whose rotating log path is fixed at import.
configure_pymmcore_process_log()

import napari

from napari_raman_widget import HardwareWidget


def _apply_windows_native_icon(qt_window, icon_path: Path) -> None:
    """Force the native HWND to use the LipidDrop icon on the Windows taskbar."""
    if sys.platform != "win32":
        return

    import ctypes

    user32 = ctypes.windll.user32
    IMAGE_ICON = 1
    LR_LOADFROMFILE = 0x0010
    WM_SETICON = 0x0080
    ICON_SMALL = 0
    ICON_BIG = 1

    hwnd = int(qt_window.winId())
    big = user32.LoadImageW(
        None, str(icon_path), IMAGE_ICON, 32, 32, LR_LOADFROMFILE
    )
    small = user32.LoadImageW(
        None, str(icon_path), IMAGE_ICON, 16, 16, LR_LOADFROMFILE
    )
    if big:
        user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, big)
    if small:
        user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, small)

    # Keep handles alive for the life of the window/process.
    qt_window._lipiddrop_icon_handles = (big, small)


if __name__ == "__main__":
    viewer = napari.Viewer()
    viewer.axes.visible = False  # napari-micromanager axes crash workaround

    from qtpy.QtCore import QTimer
    from qtpy.QtGui import QIcon
    from qtpy.QtWidgets import QApplication

    assets = WORKSPACE / "assets"
    icon_path = (
        assets / "lipid-adipocyte-taskbar.ico"
        if sys.platform == "win32"
        else assets / "lipid-adipocyte-taskbar.png"
    )
    icon = QIcon(str(icon_path))

    app = QApplication.instance()
    app.setApplicationName("LipidDrop Imaging")
    app.setApplicationDisplayName("LipidDrop Imaging")
    app.setWindowIcon(icon)

    qt_window = viewer.window._qt_window
    qt_window.setWindowTitle("LipidDrop Imaging")
    qt_window.setWindowIcon(icon)

    widget = HardwareWidget(viewer)
    viewer.window.add_dock_widget(widget, name="LipidDrop Control", area="right")

    # Napari/Qt may update the native window during startup, so apply the native
    # taskbar icon once now and once again after the event loop begins.
    _apply_windows_native_icon(qt_window, icon_path)
    QTimer.singleShot(
        0, lambda: _apply_windows_native_icon(qt_window, icon_path)
    )

    napari.run()

