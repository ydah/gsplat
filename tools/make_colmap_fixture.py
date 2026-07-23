#!/usr/bin/env python3
"""Generate matching tiny COLMAP text and binary sparse models."""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path


CAMERAS = [
    (1, 0, "SIMPLE_PINHOLE", 640, 480, [500.0, 320.0, 240.0]),
    (2, 1, "PINHOLE", 640, 480, [510.0, 515.0, 320.0, 240.0]),
    (3, 4, "OPENCV", 640, 480, [520.0, 525.0, 320.0, 240.0, 0.01, -0.02, 0.001, -0.002]),
]
IMAGES = [
    (1, [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1, "camera_000.png"),
    (2, [math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0], [1.0, 2.0, 3.0], 2, "camera_001.png"),
    (3, [1.0, 0.0, 0.0, 0.0], [-1.0, 0.5, 2.0], 3, "camera_002.png"),
]


def points():
    for index in range(100):
        yield (
            1000 + index,
            [index * 0.01, (index % 10) * 0.02, 1.0 + index * 0.001],
            [index % 256, (index * 3) % 256, (index * 7) % 256],
            0.1 + index * 0.001,
        )


def observations(image_id):
    return [
        ((index % 10) * 10.0 + image_id, (index // 10) * 10.0 + image_id, 1000 + index)
        for index in range(100)
    ]


def write_text(directory: Path):
    camera_lines = ["# Camera list"]
    for camera_id, _, model, width, height, params in CAMERAS:
        camera_lines.append(f"{camera_id} {model} {width} {height} " + " ".join(map(str, params)))
    (directory / "cameras.txt").write_text("\n".join(camera_lines) + "\n", encoding="utf-8")

    image_lines = ["# Image list"]
    for image_id, qvec, tvec, camera_id, name in IMAGES:
        image_lines.append(" ".join(map(str, [image_id, *qvec, *tvec, camera_id, name])))
        image_lines.append(" ".join(str(value) for item in observations(image_id) for value in item))
    (directory / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")

    point_lines = ["# 3D point list"]
    for point_id, xyz, rgb, error in points():
        track = [value for image_id in range(1, 4) for value in (image_id, point_id - 1000)]
        point_lines.append(" ".join(map(str, [point_id, *xyz, *rgb, error, *track])))
    (directory / "points3D.txt").write_text("\n".join(point_lines) + "\n", encoding="utf-8")


def write_binary(directory: Path):
    with (directory / "cameras.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", len(CAMERAS)))
        for camera_id, model_id, _, width, height, params in CAMERAS:
            stream.write(struct.pack("<iiQQ", camera_id, model_id, width, height))
            stream.write(struct.pack(f"<{len(params)}d", *params))

    with (directory / "images.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", len(IMAGES)))
        for image_id, qvec, tvec, camera_id, name in IMAGES:
            stream.write(struct.pack("<i4d3di", image_id, *qvec, *tvec, camera_id))
            stream.write(name.encode("utf-8") + b"\0")
            items = observations(image_id)
            stream.write(struct.pack("<Q", len(items)))
            for x_coord, y_coord, point_id in items:
                stream.write(struct.pack("<ddQ", x_coord, y_coord, point_id))

    point_list = list(points())
    with (directory / "points3D.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", len(point_list)))
        for point_id, xyz, rgb, error in point_list:
            stream.write(struct.pack("<Q3d3BdQ", point_id, *xyz, *rgb, error, 3))
            for image_id in range(1, 4):
                stream.write(struct.pack("<ii", image_id, point_id - 1000))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("test/fixtures/colmap"))
    args = parser.parse_args()
    for name in ("text", "binary"):
        directory = args.out / name
        directory.mkdir(parents=True, exist_ok=True)
        (write_text if name == "text" else write_binary)(directory)


if __name__ == "__main__":
    main()
