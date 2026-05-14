import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import random
import re

import cv2
import numpy as np

ROTATION_RE = re.compile(
    r"(?:^|_)rx(?P<rx>-?\d+(?:\.\d+)?)_"
    r"ry(?P<ry>-?\d+(?:\.\d+)?)_"
    r"rz(?P<rz>-?\d+(?:\.\d+)?)(?:_|$)"
)


@dataclass(frozen=True)
class RotationAngles:
    rx: float
    ry: float
    rz: float

    def as_dict(self):
        return {"X": self.rx, "Y": self.ry, "Z": self.rz}


def parse_rotation_from_filename(path: str | Path) -> RotationAngles:
    """
    Extracts angles from names like view_rx-40.0_ry0.0_rz40.0.png.
    """
    match = ROTATION_RE.search(Path(path).stem)
    if not match:
        raise ValueError(f"Не удалось извлечь rx/ry/rz из имени файла: {path}")

    return RotationAngles(
        rx=float(match.group("rx")),
        ry=float(match.group("ry")),
        rz=float(match.group("rz")),
    )


def calculate_optimal_focal_length(image_width: int, image_height: int, angles: RotationAngles):
    tilt_angle = max(abs(angles.rx), abs(angles.ry))
    angle_rad = math.radians(tilt_angle)
    max_dimension = max(image_width, image_height)
    base_focal_length = max_dimension * (1.0 + math.sin(angle_rad))
    return max(1, int(round(base_focal_length / 100.0) * 100))


def upscale_bilinear(input_img: np.ndarray, scale_factor: float):
    if scale_factor <= 0:
        raise ValueError("--upscale должен быть больше 0")

    if scale_factor == 1:
        return input_img

    new_width = int(input_img.shape[1] * scale_factor)
    new_height = int(input_img.shape[0] * scale_factor)
    return cv2.resize(input_img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)


def center_crop_to_aspect(input_img: np.ndarray, target_width: int, target_height: int):
    height, width = input_img.shape[:2]
    target_aspect = target_width / target_height
    image_aspect = width / height

    if image_aspect > target_aspect:
        crop_width = int(height * target_aspect)
        x0 = (width - crop_width) // 2
        x1 = x0 + crop_width
        return input_img[:, x0:x1]

    crop_height = int(width / target_aspect)
    y0 = (height - crop_height) // 2
    y1 = y0 + crop_height
    return input_img[y0:y1, :]


def rotation_matrix(axis: str, angle_deg: float):
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)

    if axis == "X":
        return np.array(
            [
                [1, 0, 0, 0],
                [0, c, -s, 0],
                [0, s, c, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

    if axis == "Y":
        return np.array(
            [
                [c, 0, s, 0],
                [0, 1, 0, 0],
                [-s, 0, c, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

    return np.array(
        [
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )


def compose_rotation_matrix(angles: RotationAngles, order: str = "ZXY"):
    matrix = np.eye(4, dtype=np.float32)
    axis_angles = angles.as_dict()

    for axis in order.upper():
        matrix = rotation_matrix(axis, axis_angles[axis]) @ matrix

    return matrix


def perspective_transform_matrix(
    image_width: int,
    image_height: int,
    angles: RotationAngles,
    order: str = "ZXY",
):
    focal_length = calculate_optimal_focal_length(image_width, image_height, angles)
    output_scale = 2.0 + max(abs(angles.rx), abs(angles.ry)) / 45.0
    output_width = int(image_width * output_scale)
    output_height = int(image_height * output_scale)

    proj_2d_to_3d = np.array(
        [
            [1, 0, -(image_width / 2.0)],
            [0, 1, -(image_height / 2.0)],
            [0, 0, 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    trans = np.eye(4, dtype=np.float32)
    trans[2, 3] = focal_length

    proj_3d_to_2d = np.array(
        [
            [focal_length, 0, output_width / 2.0, 0],
            [0, focal_length, output_height / 2.0, 0],
            [0, 0, 1, 0],
        ],
        dtype=np.float32,
    )

    rotation = compose_rotation_matrix(angles, order=order)
    return proj_3d_to_2d @ trans @ rotation @ proj_2d_to_3d, (output_width, output_height)


def rotate_background(
    background: np.ndarray,
    angles: RotationAngles,
    target_size: tuple[int, int],
    order: str = "ZXY",
    upscale: float = 2.0,
):
    target_width, target_height = target_size
    work_img = upscale_bilinear(background, upscale)
    work_height, work_width = work_img.shape[:2]
    transform, output_size = perspective_transform_matrix(
        image_width=work_width,
        image_height=work_height,
        angles=angles,
        order=order,
    )

    rotated = cv2.warpPerspective(
        work_img,
        transform,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    cropped = center_crop_to_aspect(rotated, target_width, target_height)
    return cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)


def list_images(directory: Path, pattern: str):
    return sorted(
        path
        for path in directory.glob(pattern)
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )


def choose_background(backgrounds: list[Path], index: int, rng: random.Random, selection: str):
    if selection == "random":
        return rng.choice(backgrounds)
    return backgrounds[index % len(backgrounds)]


def target_size_from_image(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Не удалось открыть файл мины: {path}")

    height, width = image.shape[:2]
    return width, height


def rotate_backgrounds_for_mines(
    backgrounds_dir: Path,
    mines_dir: Path,
    outdir: Path,
    background_glob: str = "*.png",
    mine_glob: str = "*.png",
    order: str = "ZXY",
    upscale: float = 2.0,
    selection: str = "cycle",
    seed: int | None = None,
    limit: int | None = None,
):
    backgrounds = list_images(backgrounds_dir, background_glob)
    mines = list_images(mines_dir, mine_glob)

    if not backgrounds:
        raise FileNotFoundError(f"Не найдены фоны: {backgrounds_dir}/{background_glob}")
    if not mines:
        raise FileNotFoundError(f"Не найдены изображения мины: {mines_dir}/{mine_glob}")

    outdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    processed = 0

    for index, mine_path in enumerate(mines):
        if limit is not None and processed >= limit:
            break

        angles = parse_rotation_from_filename(mine_path)
        background_path = choose_background(backgrounds, index, rng, selection)
        background = cv2.imread(str(background_path), cv2.IMREAD_COLOR)
        if background is None:
            raise FileNotFoundError(f"Не удалось открыть фон: {background_path}")

        rotated = rotate_background(
            background=background,
            angles=angles,
            target_size=target_size_from_image(mine_path),
            order=order,
            upscale=upscale,
        )

        output_path = outdir / mine_path.name
        cv2.imwrite(str(output_path), rotated)
        processed += 1
        print(
            f"[{processed:>5}/{len(mines)}] {background_path.name} -> {output_path.name} "
            f"(rx={angles.rx}, ry={angles.ry}, rz={angles.rz})"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Rotate generated backgrounds using rx/ry/rz from mine render filenames."
    )
    parser.add_argument("--backgrounds-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--mines-dir", type=Path, default=Path("data/pfm"))
    parser.add_argument("--outdir", type=Path, default=Path("data/interim/backgrounds_rotated"))
    parser.add_argument("--background-glob", default="*.png")
    parser.add_argument("--mine-glob", default="*.png")
    parser.add_argument(
        "--order",
        default="ZXY",
        choices=["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"],
        help="Rotation order; keep it equal to render_stl_euler_sweep.py --order.",
    )
    parser.add_argument("--upscale", type=float, default=2.0)
    parser.add_argument("--selection", choices=["cycle", "random"], default="cycle")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rotate_backgrounds_for_mines(
        backgrounds_dir=args.backgrounds_dir,
        mines_dir=args.mines_dir,
        outdir=args.outdir,
        background_glob=args.background_glob,
        mine_glob=args.mine_glob,
        order=args.order,
        upscale=args.upscale,
        selection=args.selection,
        seed=args.seed,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
