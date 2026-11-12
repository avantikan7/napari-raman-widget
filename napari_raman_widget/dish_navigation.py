"""Dish field navigation for repeated live-cell imaging.

Saved fields keep stage XYZ plus an optional reference image. Stage motion is
always explicit: only "Go to selected field" moves XY. Image re-registration
is computational only and never commands microscope hardware.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

REGISTERED_LAYER = "dish field registered preview"


@dataclass
class DishField:
    name: str
    x_um: float
    y_um: float
    z_um: float
    reference_image: np.ndarray | None = None


def image_projection(data):
    """Return a finite 2D image suitable for field registration."""
    image = np.asarray(data, dtype=np.float32)
    if image.ndim == 2:
        result = image
    elif image.ndim == 3:
        result = np.max(image, axis=0)
    else:
        raise ValueError(f"need a 2D image or Z-Y-X stack, got {image.shape}")
    if not np.isfinite(result).all():
        raise ValueError("registration image contains non-finite pixels")
    return result


def register_images(reference, current):
    """Return shift (dy, dx) that aligns current to reference."""
    from skimage.registration import phase_cross_correlation

    reference = image_projection(reference)
    current = image_projection(current)
    if reference.shape != current.shape:
        raise ValueError(
            f"reference shape {reference.shape} != current shape {current.shape}"
        )
    shift, error, _phase = phase_cross_correlation(
        reference, current, upsample_factor=10
    )
    shift = np.asarray(shift, dtype=float)
    if shift.shape != (2,) or not np.isfinite(shift).all():
        raise ValueError("registration returned an invalid shift")
    return shift, float(error)


def shifted_preview(current, shift_yx):
    """Return current translated by the registration shift for visual QC."""
    from scipy.ndimage import shift

    return shift(
        image_projection(current),
        shift=np.asarray(shift_yx, dtype=float),
        order=1,
        mode="constant",
        cval=0.0,
    )


class DishFieldNavigationPanel(QWidget):
    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.fields: list[DishField] = []

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Save named stage fields in a dish and revisit them during a live "
            "time course. Stage motion occurs only when you explicitly press "
            "'Go to selected field'."
        )
        intro.setWordWrap(True)
        intro.setToolTip(
            "A saved field stores stage X/Y/Z and, when a grayscale image layer "
            "is selected, a 2D registration reference. Re-registration itself "
            "does not move the microscope."
        )
        layout.addWidget(intro)

        self.save_button = QPushButton("Save current field")
        self.save_button.setToolTip(
            "Read the current stage XYZ and save it as Field N. If a grayscale "
            "2D or Z-Y-X image layer is selected, its projection is also stored "
            "as the registration reference for this field."
        )
        self.save_button.clicked.connect(self.save_current_field)
        layout.addWidget(self.save_button)

        self.list = QListWidget()
        self.list.setToolTip(
            "Saved dish fields for this Napari session. Select one before "
            "navigation or image re-registration."
        )
        layout.addWidget(self.list)

        row = QHBoxLayout()
        self.go_button = QPushButton("Go to selected field")
        self.go_button.setToolTip(
            "Explicit hardware action: move the XY stage to the saved field. "
            "Saved Z is shown as a reference but is not commanded; refocus or "
            "use MDA autofocus as appropriate."
        )
        self.register_button = QPushButton(
            "Re-register current image to saved field"
        )
        self.register_button.setToolTip(
            "Software only: phase-correlate the selected image against this "
            "field's saved reference image, report dy/dx drift, and create an "
            "aligned preview layer. This does NOT move the stage."
        )
        row.addWidget(self.go_button)
        row.addWidget(self.register_button)
        layout.addLayout(row)

        self.go_button.clicked.connect(self.go_to_selected_field)
        self.register_button.clicked.connect(self.reregister_current_image)

        self.status = QLabel("No dish fields saved.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

    def _selected_field(self):
        row = self.list.currentRow()
        if row < 0 or row >= len(self.fields):
            raise ValueError("Select a saved field first")
        return self.fields[row]

    def _selected_reference_image(self):
        layer = self.owner.viewer.layers.selection.active
        if layer is None or not hasattr(layer, "data"):
            return None
        try:
            return image_projection(layer.data).copy()
        except ValueError:
            return None

    def save_current_field(self):
        core = self.owner.core
        try:
            if core is None or not core.getXYStageDevice():
                raise RuntimeError("Connect the XY stage before saving a field")
            if core.mda.is_running() or getattr(
                self.owner, "_raman_mda_pending", False
            ):
                raise RuntimeError("Wait for acquisition to finish before saving")
            x, y = core.getXYPosition()
            z = core.getPosition()
            values = np.asarray([x, y, z], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("Current stage XYZ is not finite")
            name = f"Field {len(self.fields) + 1}"
            reference = self._selected_reference_image()
            self.fields.append(
                DishField(name, float(x), float(y), float(z), reference)
            )
            self.list.addItem(
                f"{name}   X={x:.1f}  Y={y:.1f}  Z={z:.1f} um"
                + ("   [image]" if reference is not None else "")
            )
            self.list.setCurrentRow(len(self.fields) - 1)
            self.status.setText(
                f"Saved {name}. "
                + (
                    "Reference image stored for registration."
                    if reference is not None
                    else "No suitable image layer was selected; stage position only."
                )
            )
        except Exception as error:
            self.status.setText(f"Could not save field: {error}")

    def go_to_selected_field(self):
        try:
            field = self._selected_field()
            core = self.owner.core
            if core is None or not core.getXYStageDevice():
                raise RuntimeError("Connect the XY stage before navigating")
            if core.mda.is_running() or getattr(
                self.owner, "_raman_mda_pending", False
            ):
                raise RuntimeError("Wait for acquisition to finish before navigating")
            core.setXYPosition(field.x_um, field.y_um)
            core.waitForSystem()
            self.status.setText(
                f"Moved XY to {field.name}: ({field.x_um:.1f}, "
                f"{field.y_um:.1f}) um. Saved Z={field.z_um:.1f} um was NOT moved."
            )
        except Exception as error:
            self.status.setText(f"Could not navigate: {error}")

    def reregister_current_image(self):
        try:
            field = self._selected_field()
            if field.reference_image is None:
                raise ValueError(
                    "This field has no saved reference image. Re-save it while "
                    "the desired image layer is selected."
                )
            current = self._selected_reference_image()
            if current is None:
                raise ValueError("Select a grayscale 2D or Z-Y-X image layer first")
            shift_yx, error = register_images(field.reference_image, current)
            preview = shifted_preview(current, shift_yx)
            if REGISTERED_LAYER in self.owner.viewer.layers:
                self.owner.viewer.layers[REGISTERED_LAYER].data = preview
            else:
                self.owner.viewer.add_image(
                    preview,
                    name=REGISTERED_LAYER,
                    colormap="gray",
                    blending="additive",
                )
            self.status.setText(
                f"{field.name} registration: dy={shift_yx[0]:+.2f} px, "
                f"dx={shift_yx[1]:+.2f} px; phase-correlation error={error:.3g}. "
                "Preview aligned; stage did not move."
            )
        except Exception as error:
            self.status.setText(f"Could not re-register image: {error}")
