from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from napari_raman_widget.dish_navigation import (
    DishField,
    image_projection,
    register_images,
    shifted_preview,
)


def disk(shape, center, radius):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= radius ** 2


def test_image_projection_accepts_2d_and_z_stack():
    image = np.zeros((32, 32), dtype=np.float32)
    image[10, 12] = 2
    assert np.array_equal(image_projection(image), image)

    stack = np.stack([image, image * 3])
    assert np.array_equal(image_projection(stack), image * 3)


def test_registration_recovers_whole_field_translation():
    from scipy import ndimage as ndi

    reference = np.zeros((96, 96), dtype=np.float32)
    reference[disk(reference.shape, (30, 35), 6)] = 1.0
    reference[disk(reference.shape, (65, 60), 8)] = 0.7
    current = ndi.shift(reference, shift=(8, -6), order=1, mode="constant")

    shift, error = register_images(reference, current)

    np.testing.assert_allclose(shift, [-8, 6], atol=0.25)
    assert np.isfinite(error)

    aligned = shifted_preview(current, shift)
    assert aligned.shape == reference.shape


def test_saved_field_is_plain_state_not_hardware_action():
    field = DishField(
        "Field 1",
        x_um=100.0,
        y_um=200.0,
        z_um=50.0,
        reference_image=np.ones((10, 10), dtype=np.float32),
    )
    assert field.name == "Field 1"
    assert field.x_um == 100.0
    assert field.y_um == 200.0
    assert field.z_um == 50.0
    assert field.reference_image.shape == (10, 10)
