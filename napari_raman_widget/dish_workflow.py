"""Two-pass dish workflow: segment every imaging tile, review, then aim Raman.

Pure functions, no hardware and no Qt. Pass 1 turns each tile image into
droplet targets (pixel coordinates). Pass 2 maps those targets to absolute
stage micrometers, drops duplicates seen in overlapping tiles, and returns
the list a Raman grid/point acquisition can consume.
"""
import csv

import numpy as np


def tile_centers(center_xy, radius_um, fov_um, overlap=0.1):
    """Stage (x, y) um of square FOV tiles that cover the circular well.

    A tile is kept when any part of it touches the circle, ordered as a
    serpentine so the stage moves short distances.
    """
    fov = np.broadcast_to(np.asarray(fov_um, float), (2,))
    step = fov * (1 - overlap)
    if not (0 <= overlap < 1) or np.any(fov <= 0) or radius_um <= 0:
        raise ValueError("Need 0 <= overlap < 1 and positive FOV and radius")
    nx, ny = (int(np.ceil(radius_um / s)) for s in step)
    cx, cy = center_xy
    tiles = []
    for j, iy in enumerate(range(-ny, ny + 1)):
        xs = range(-nx, nx + 1) if j % 2 == 0 else range(nx, -nx - 1, -1)
        for ix in xs:
            x, y = cx + ix * step[0], cy + iy * step[1]
            # Nearest point of the tile rectangle to the well center.
            dx = max(abs(x - cx) - fov[0] / 2, 0)
            dy = max(abs(y - cy) - fov[1] / 2, 0)
            if np.hypot(dx, dy) <= radius_um:
                tiles.append((float(x), float(y)))
    return tiles


def pixel_to_stage(yx, tile_xy, image_shape, pixel_um, axes=((1, 0), (0, 1))):
    """Map (row, col) pixels of a tile to stage (x, y) um.

    ``tile_xy`` is the stage position at the image center. ``axes`` is the
    2x2 matrix taking (col offset, row offset) to (dx, dy) in pixels' stage
    directions; flip signs here for reversed stage axes. Replace with the
    measured grid calibration when the real hardware is connected.
    """
    yx = np.asarray(yx, float).reshape(-1, 2)
    center = (np.asarray(image_shape[-2:], float) - 1) / 2
    offset = (yx - center)[:, ::-1]  # (col, row) = (x, y) pixels
    return np.asarray(tile_xy, float) + pixel_um * offset @ np.asarray(axes, float)


def segment_tiles(tiles, images, segment_fn):
    """Pass 1. ``segment_fn(image) -> list of droplet target dicts``.

    Returns one record per tile with its targets, for review before any
    Raman time is spent.
    """
    if len(tiles) != len(images):
        raise ValueError("Need one image per tile")
    return [{"tile": i, "tile_xy": tuple(xy), "shape": np.shape(img)[-2:],
             "targets": segment_fn(img)}
            for i, (xy, img) in enumerate(zip(tiles, images))]


def stage_targets(records, pixel_um, axes=((1, 0), (0, 1)),
                  min_separation_um=None, include_grid=False):
    """Pass 2. Absolute stage points for every reviewed droplet.

    A droplet imaged in two overlapping tiles is kept once (the copy farther
    from its tile edge wins). ``min_separation_um`` defaults to the smallest
    droplet radius seen, so touching droplets are not merged.
    """
    found = []
    for rec in records:
        h, w = rec["shape"]
        for t in rec["targets"]:
            aim = pixel_to_stage(t["aim"], rec["tile_xy"], rec["shape"], pixel_um, axes)[0]
            edge = min(t["aim"][0], h - 1 - t["aim"][0], t["aim"][1], w - 1 - t["aim"][1])
            grid = (pixel_to_stage(t["grid"], rec["tile_xy"], rec["shape"], pixel_um, axes)
                    if include_grid and len(t.get("grid", [])) else np.empty((0, 2)))
            found.append({"tile": rec["tile"], "xy": aim, "edge": edge,
                          "radius_um": float(t["radius"]) * pixel_um, "grid": grid})
    if not found:
        return []
    sep = (min_separation_um if min_separation_um is not None
           else min(f["radius_um"] for f in found))
    kept = []
    for f in sorted(found, key=lambda f: -f["edge"]):
        if all(np.hypot(*(f["xy"] - k["xy"])) > sep for k in kept):
            kept.append(f)
    return sorted(kept, key=lambda f: (f["tile"], f["xy"][1], f["xy"][0]))


def export_targets_csv(path, targets):
    """Review table: one row per droplet in absolute stage um."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        out = csv.writer(fh)
        out.writerow(["tile", "stage_x_um", "stage_y_um", "radius_um", "grid_points"])
        for t in targets:
            out.writerow([t["tile"], f"{t['xy'][0]:.2f}", f"{t['xy'][1]:.2f}",
                          f"{t['radius_um']:.2f}", len(t["grid"])])
