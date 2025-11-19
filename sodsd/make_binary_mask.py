import os

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
    mask = cv2.bitwise_not(mask)

    # сохраняем маску
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, mask)


if __name__ == "__main__":
    # пример использования
    output_mask = "/home/ubuntu/sodsd/data/maskPfm"
    source_path = "/home/ubuntu/sodsd/data/pfm"

    path_images = os.listdir(source_path)
    for path in path_images:
        print(path)
        image_path = os.path.join(source_path, path)
        output_mask = os.path.join(output_mask, path)

        create_binary_mask(
            image_path=image_path,
            output_path=output_mask,
            thresh=240,  # если фон не идеально белый — можно поиграть 220–250
        )
