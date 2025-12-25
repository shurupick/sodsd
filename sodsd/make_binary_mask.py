from pathlib import Path

import cv2


def create_binary_mask(
    image_path: str,
    output_path: str,
    thresh: int = 240,
) -> None:
    """
    Создаёт бинарную маску по изображению с белым фоном.

    :param image_path: путь к исходному PNG изображению
    :param output_path: путь для сохранения маски (PNG)
    :param thresh: порог яркости (0–255). Чем ближе к 255, тем меньше шума от фона.
    """
    # читаем изображение
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Не удалось открыть файл: {image_path}")

    # переводим в градации серого
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # инверсный порог:
    #   если pixel > thresh → 0 (фон)
    #   иначе → 255 (мина/объект)
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    # mask = cv2.bitwise_not(mask)

    # сохраняем маску, гарантируем существование пути
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), mask)


if __name__ == "__main__":
    # пример использования
    source_dir = Path("./data/pfm")
    output_dir = Path("./data/maskPfm")

    if not source_dir.is_dir():
        raise SystemExit(f"Источник не найден: {source_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_suffixes = {".png", ".jpg", ".jpeg"}
    for image_path in sorted(source_dir.iterdir()):
        if not image_path.is_file():
            continue
        if image_path.name.startswith(".") or image_path.suffix.lower() not in valid_suffixes:
            continue
        output_mask = output_dir / image_path.name
        print(f"masking {image_path.name}")

        create_binary_mask(
            image_path=str(image_path),
            output_path=str(output_mask),
            thresh=240,  # если фон не идеально белый — можно поиграть 220–250
        )
