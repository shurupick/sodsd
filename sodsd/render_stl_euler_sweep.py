import argparse
import colorsys
import os
import random

import numpy as np
import pyvista as pv
from stl import mesh as npstl

PFM_GREEN_PRESETS = {
    "pfm_green": "#556b2f",
    "olive_green": "#5f6f34",
    "dark_olive": "#3f4f25",
    "moss_green": "#687b3a",
    "faded_green": "#77855a",
    "khaki_green": "#7a8051",
    "yellow_olive": "#8a8a46",
}

PFM_NATURAL_GREEN_PALETTE = tuple(PFM_GREEN_PRESETS.values())


def parse_range(spec: str):
    """
    Парсит строку вида 'start:end:step' (в градусах, включительно по end, если попадает по шагу).
    Примеры: '0:360:15', '0:90:10', '30:30:1'
    """
    a, b, s = map(float, spec.split(":"))
    if s == 0:
        raise ValueError("Шаг не может быть 0")
    # включительно конец, если попадает по сетке
    vals = np.arange(a, b + 1e-9, s)
    return [float(v) for v in vals]


def _rgb_to_float(rgb):
    return tuple(channel / 255.0 for channel in rgb)


def _hex_to_rgb_float(value: str):
    value = value.lstrip("#")
    if len(value) != 6:
        raise ValueError("HEX-цвет должен иметь формат #RRGGBB")
    return _rgb_to_float(
        (
            int(value[0:2], 16),
            int(value[2:4], 16),
            int(value[4:6], 16),
        )
    )


def resolve_color(color_spec: str):
    """
    Возвращает RGB tuple в диапазоне 0..1.

    Поддерживаются:
    - #RRGGBB
    - стандартные имена цветов PyVista/VTK
    - локальные пресеты зелёных оттенков PFM_GREEN_PRESETS
    """
    key = color_spec.strip().lower().replace("-", "_")
    if key in PFM_GREEN_PRESETS:
        return _hex_to_rgb_float(PFM_GREEN_PRESETS[key])

    try:
        return tuple(pv.Color(color_spec).float_rgb)
    except ValueError as exc:
        presets = ", ".join(sorted(PFM_GREEN_PRESETS))
        raise ValueError(
            f"Неизвестный цвет '{color_spec}'. Используйте #RRGGBB, имя цвета "
            f"PyVista/VTK или один из пресетов: {presets}"
        ) from exc


def jitter_color(rgb, rng: random.Random, strength: float):
    """
    Делает цвет чуть менее синтетическим: небольшой разброс в HSV вокруг базы.
    strength=0 отключает разброс, 1 даёт максимальный предусмотренный диапазон.
    """
    strength = max(0.0, min(1.0, strength))
    if strength == 0:
        return rgb

    hue, saturation, value = colorsys.rgb_to_hsv(*rgb)
    hue = (hue + rng.uniform(-0.035, 0.035) * strength) % 1.0
    saturation_scale = 1.0 + rng.uniform(-0.2, 0.12) * strength
    value_scale = 1.0 + rng.uniform(-0.22, 0.15) * strength
    saturation = min(1.0, max(0.25, saturation * saturation_scale))
    value = min(0.72, max(0.18, value * value_scale))
    return colorsys.hsv_to_rgb(hue, saturation, value)


def choose_natural_green(rng: random.Random, jitter_strength: float):
    base_color = rng.choice(PFM_NATURAL_GREEN_PALETTE)
    return jitter_color(_hex_to_rgb_float(base_color), rng, jitter_strength)


def main():
    p = argparse.ArgumentParser(description="Batch-рендер STL по углам Эйлера")
    p.add_argument("--stl", required=True, help="Путь к STL-модели")
    p.add_argument(
        "--rx",
        default="-60:0:10",
        help="Диапазон углов вокруг X, формат start:end:step (°)",
    )
    p.add_argument(
        "--ry",
        default="0:0:10",
        help="Диапазон углов вокруг Y, формат start:end:step (°)",
    )
    p.add_argument(
        "--rz",
        default="0:90:10",
        help="Диапазон углов вокруг Z, формат start:end:step (°)",
    )
    p.add_argument(
        "--order",
        default="ZXY",
        choices=["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"],
        help="Порядок применения углов Эйлера к МОДЕЛИ (по умолчанию ZYX)",
    )
    p.add_argument("--w", type=int, default=1200, help="Ширина изображения")
    p.add_argument("--h", type=int, default=900, help="Высота изображения")
    p.add_argument("--bg", default="white", help="Цвет фона (например, white/black/#RRGGBB)")
    p.add_argument(
        "--fmt", default="png", choices=["png", "jpg", "jpeg"], help="Формат сохранения"
    )
    p.add_argument("--outdir", default="./data/pfm", help="Папка для сохранения")
    p.add_argument("--name", default="view", help="Префикс имени файлов")
    p.add_argument(
        "--color",
        default="pfm_green",
        help=(
            "Цвет модели: #RRGGBB, имя цвета PyVista/VTK или пресет "
            f"({', '.join(sorted(PFM_GREEN_PRESETS))})"
        ),
    )
    p.add_argument(
        "--randomize-color",
        choices=["off", "once", "per-frame"],
        default="off",
        help=(
            "Рандомизация натурального зелёного оттенка: off — использовать --color, "
            "once — один оттенок на весь запуск, per-frame — новый оттенок на каждый кадр"
        ),
    )
    p.add_argument(
        "--color-jitter",
        type=float,
        default=0.35,
        help="Сила естественного разброса HSV для --randomize-color (0..1)",
    )
    p.add_argument(
        "--color-seed",
        type=int,
        help="Seed для воспроизводимого выбора оттенков",
    )
    p.add_argument("--smooth", action="store_true", help="Включить сглаженное освещение (Phong)")
    p.add_argument("--aa", type=int, default=8, help="MSAA (кол-во выборок, 0=выкл)")
    p.add_argument(
        "--camera",
        default="top",
        choices=["isometric", "top", "front", "right", "custom"],
        help="Базовая позиция камеры (камера фиксирована, вращаем модель)",
    )
    p.add_argument(
        "--cam_roll",
        type=float,
        default=0.0,
        help="Поворот камеры (roll, °) при camera=custom",
    )
    p.add_argument(
        "--cam_pos",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Позиция камеры при camera=custom",
    )
    p.add_argument(
        "--cam_focal",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Точка фокуса камеры при camera=custom",
    )
    p.add_argument(
        "--cam_up",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Вектор up камеры при camera=custom",
    )
    args = p.parse_args()
    rng = random.Random(args.color_seed)

    os.makedirs(args.outdir, exist_ok=True)

    # Загружаем STL (numpy-stl) и конвертируем в PyVista PolyData
    stl_m = npstl.Mesh.from_file(args.stl)  # (n_faces, 3, 3)
    triangles = stl_m.vectors.reshape(-1, 3)  # все вершины подряд
    # Уникальные вершины + индексы (восстановим треугольники)
    uniq, inv = np.unique(np.round(triangles, 8), axis=0, return_inverse=True)
    faces = inv.reshape(-1, 3)
    # PyVista ожидает формат faces: [3, i, j, k, 3, i, j, k, ...]
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]).ravel()

    mesh = pv.PolyData(uniq, faces_pv)

    # Нормализация масштаба (чтобы кадры были сопоставимы)
    # Центрируем и приводим к единичному размеру по диагонали bbox.
    bounds = np.array(mesh.bounds).reshape(3, 2)  # [[xmin,xmax],[ymin,ymax],[zmin,zmax]]
    center = bounds.mean(axis=1)
    extent = bounds[:, 1] - bounds[:, 0]
    diag = float(np.linalg.norm(extent))
    if diag > 0:
        mesh.translate(-center, inplace=True)
        mesh.scale(1.0 / diag, inplace=True)

    pl = pv.Plotter(off_screen=True, window_size=[args.w, args.h])
    # цвет фона может быть любым — при прозрачной записи альфа обрежет его
    pl.set_background(args.bg)

    if args.aa > 0:
        try:
            pl.enable_anti_aliasing(args.aa)
        except Exception:
            pass

    # Настройка камеры
    if args.camera == "isometric":
        pl.camera_position = "iso"
    elif args.camera == "top":
        pl.camera_position = "xy"  # сверху
    elif args.camera == "front":
        pl.camera_position = "xz"  # фронт
    elif args.camera == "right":
        pl.camera_position = "yz"  # справа
    elif args.camera == "custom":
        # Пользовательская позиция
        if args.cam_pos and args.cam_focal and args.cam_up:
            pl.camera.position = tuple(args.cam_pos)
            pl.camera.focal_point = tuple(args.cam_focal)
            pl.camera.up = tuple(args.cam_up)
        if args.cam_roll:
            pl.camera.roll = args.cam_roll

    if args.randomize_color == "off":
        initial_color = resolve_color(args.color)
    else:
        initial_color = choose_natural_green(rng, args.color_jitter)

    # Базовая сетка без вращения (чтобы переиспользовать актёр)
    actor = pl.add_mesh(mesh, color=initial_color, smooth_shading=args.smooth, specular=0.12)
    pl.camera.zoom(0.5)

    Rx = parse_range(args.rx)
    Ry = parse_range(args.ry)
    Rz = parse_range(args.rz)

    # Чтобы быстрее, создаём копию и вращаем INPLACE на каждом шаге относительно исходной геометрии
    base = mesh.copy(deep=True)

    order = args.order.upper()
    n_total = len(Rx) * len(Ry) * len(Rz)
    idx = 0

    def _rotation_matrix(axis: str, angle_deg: float):
        theta = np.deg2rad(angle_deg)
        c, s = np.cos(theta), np.sin(theta)
        if axis == "X":
            return np.array(
                [
                    [1, 0, 0, 0],
                    [0, c, -s, 0],
                    [0, s, c, 0],
                    [0, 0, 0, 1],
                ],
                dtype=float,
            )
        if axis == "Y":
            return np.array(
                [
                    [c, 0, s, 0],
                    [0, 1, 0, 0],
                    [-s, 0, c, 0],
                    [0, 0, 0, 1],
                ],
                dtype=float,
            )
        return np.array(
            [
                [c, -s, 0, 0],
                [s, c, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )

    for rx in Rx:
        for ry in Ry:
            for rz in Rz:
                idx += 1
                if args.randomize_color == "per-frame":
                    actor.GetProperty().SetColor(choose_natural_green(rng, args.color_jitter))

                mesh_copy = base.copy(deep=True)
                axis_angles = {"X": rx, "Y": ry, "Z": rz}
                matrix = np.eye(4, dtype=float)
                for ax in order:
                    matrix = _rotation_matrix(ax, axis_angles[ax]) @ matrix

                mesh_copy.transform(matrix, inplace=True)
                actor.mapper.SetInputData(mesh_copy)
                pl.render()

                # рендер и сохранение
                out_name = f"{args.name}_rx{rx:.1f}_ry{ry:.1f}_rz{rz:.1f}.{args.fmt}"
                out_path = os.path.join(args.outdir, out_name)

                try:
                    pl.screenshot(out_path, transparent_background=True)
                except TypeError:
                    # fallback для старых версий (если есть 'transparent')
                    try:
                        pl.screenshot(out_path, transparent_background=True)
                    except TypeError:
                        # совсем старый вариант — сохраним как есть, но фон уже будет просто цветным
                        pl.screenshot(out_path)

                print(f"[{idx:>5}/{n_total}] saved: {out_path}")

    pl.close()


if __name__ == "__main__":
    main()
