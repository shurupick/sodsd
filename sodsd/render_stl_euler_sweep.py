# render_stl_euler_sweep.py
import argparse
import os

import numpy as np
import pyvista as pv
from stl import mesh as npstl


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


def main():
    p = argparse.ArgumentParser(description="Batch-рендер STL по углам Эйлера")
    p.add_argument("--stl", required=True, help="Путь к STL-модели")
    p.add_argument(
        "--rx",
        default="0:50:10",
        help="Диапазон углов вокруг X, формат start:end:step (°)",
    )
    p.add_argument(
        "--ry",
        default="0:0:1",
        help="Диапазон углов вокруг Y, формат start:end:step (°)",
    )
    p.add_argument(
        "--rz",
        default="0:180:30",
        help="Диапазон углов вокруг Z, формат start:end:step (°)",
    )
    p.add_argument(
        "--order",
        default="ZYX",
        choices=["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"],
        help="Порядок применения углов Эйлера к МОДЕЛИ (по умолчанию ZYX)",
    )
    p.add_argument("--w", type=int, default=1200, help="Ширина изображения")
    p.add_argument("--h", type=int, default=900, help="Высота изображения")
    p.add_argument(
        "--bg", default="white", help="Цвет фона (например, white/black/#RRGGBB)"
    )
    p.add_argument(
        "--fmt", default="png", choices=["png", "jpg", "jpeg"], help="Формат сохранения"
    )
    p.add_argument("--outdir", default="renders", help="Папка для сохранения")
    p.add_argument("--name", default="view", help="Префикс имени файлов")
    p.add_argument("--color", default="#b0c4de", help="Цвет модели")
    p.add_argument(
        "--smooth", action="store_true", help="Включить сглаженное освещение (Phong)"
    )
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

    os.makedirs(args.outdir, exist_ok=True)

    # Загружаем STL (numpy-stl) и конвертируем в PyVista PolyData
    stl_m = npstl.Mesh.from_file(args.stl)  # (n_faces, 3, 3)
    triangles = stl_m.vectors.reshape(-1, 3)  # все вершины подряд
    # Уникальные вершины + индексы (восстановим треугольники)
    uniq, inv = np.unique(np.round(triangles, 8), axis=0, return_inverse=True)
    faces = inv.reshape(-1, 3)
    # PyVista ожидает формат faces: [3, i, j, k, 3, i, j, k, ...]
    faces_pv = np.hstack(
        [np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]
    ).ravel()

    mesh = pv.PolyData(uniq, faces_pv)

    # Нормализация масштаба (чтобы кадры были сопоставимы)
    # Центрируем и приводим к единичному размеру по диагонали bbox.
    bounds = np.array(mesh.bounds).reshape(
        3, 2
    )  # [[xmin,xmax],[ymin,ymax],[zmin,zmax]]
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

    # Базовая сетка без вращения (чтобы переиспользовать актёр)
    actor = pl.add_mesh(
        mesh, color=args.color, smooth_shading=args.smooth, specular=0.2
    )
    pl.camera.zoom(0.5)

    Rx = parse_range(args.rx)
    Ry = parse_range(args.ry)
    Rz = parse_range(args.rz)

    # Чтобы быстрее, создаём копию и вращаем INPLACE на каждом шаге относительно исходной геометрии
    base = mesh.copy(deep=True)

    order = args.order.upper()
    n_total = len(Rx) * len(Ry) * len(Rz)
    idx = 0

    for rx in Rx:
        for ry in Ry:
            for rz in Rz:
                idx += 1
                # восстановим базовую геометрию на каждой итерации
                mesh_copy = base.copy(deep=True)
                # применяем углы к МОДЕЛИ в заданном порядке
                for ax, angle in zip(
                    order, [locals()[f"r{ax.lower()}"] for ax in order]
                ):
                    if ax == "X":
                        mesh_copy.rotate_x(angle, inplace=True)
                    elif ax == "Y":
                        mesh_copy.rotate_y(angle, inplace=True)
                    elif ax == "Z":
                        mesh_copy.rotate_z(angle, inplace=True)

                # обновляем актёр (меняем ссылку на геометрию)
                actor.mapper.SetInputData(mesh_copy)  # low-level VTK обновление

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
