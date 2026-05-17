from PIL import Image, ImageEnhance
import numpy as np
from scipy.ndimage import label, binary_dilation, find_objects
from ultralytics import YOLO
from pathlib import Path
from scipy.ndimage import binary_dilation, gaussian_filter


# Функция для обнаружения подписей с помощью YOLOv8 и обрезки изображений
def YOLOv8_detect(image_path: str, # путь к изображению
                  model_path: str = None, 
                  padding: int = 30 # увеличение облости box
                  ) -> list:
    
    if model_path is None:
        dir = Path(__file__).parent
        model_path = dir / 'detector_yolo_1cls.pt'

    model = YOLO(model_path) # загружаем модель
    image = Image.open(image_path).convert("RGB")
    results = model(image_path) # применяем модель к изображению

    for result in results:
        
        img_w, img_h = image.size # Получаем размеры изображения чтобы не выходить за границы при расширении bbox
        signatures = [] # Список для хранения обрезанных изображений

        for box in result.boxes.xyxy:  # [x1, y1, x2, y2] - координаты углов bbox
            x1, y1, x2, y2 = box.tolist() # Преобразуем в список для удобства
            
            # Расширяем bbox, не выходя за границы изображения
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(img_w, x2 + padding)
            y2 = min(img_h, y2 + padding)
            
            cropped = image.crop((x1, y1, x2, y2))
            signatures.append(cropped)
    return signatures




# Функция для создания маски на основе синевы и удаления фона
def mask(signatures: list) -> list:

    cleared_ = [] # Список для хранения очищенных изображений

    for image in signatures:

        img = ImageEnhance.Color(image).enhance(2.0) # Увеличиваем насыщенность синих цветов
        pict_array = np.array(img) # Преобразуем изображение в массив

        # Извлекаем каналы RGB
        red = pict_array[:, :, 0]
        green = pict_array[:, :, 1]
        blue = pict_array[:, :, 2]

        # Нормализуем каналы
        sum_rgb = red.astype(np.uint32) + green.astype(np.uint32) + blue.astype(np.uint32)
        r_norm = np.divide(red, sum_rgb, where=(sum_rgb != 0), out=np.zeros_like(red, dtype=float))
        g_norm = np.divide(green, sum_rgb, where=(sum_rgb != 0), out=np.zeros_like(red, dtype=float))
        b_norm = np.divide(blue, sum_rgb, where=(sum_rgb != 0), out=np.zeros_like(red, dtype=float))

        bg_diff = b_norm - g_norm # индекс синевы

        # безопасное деление
        BI = np.divide(
            bg_diff,
            r_norm,
            where=(r_norm != 0),
            out=np.zeros_like(bg_diff, dtype=float)
        )

        BI[BI > 1] = 1.0 # ограничим значения (по желанию)

        binary = BI > 0.5 #  (можно настроить)

        dilated = binary_dilation(binary, iterations=8) # (можно настроить количество итераций)

        # Фильтруем по размеру
        labeled, _ = label(dilated)
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        mask = np.isin(labeled, np.where(sizes > 500))

        # Возвращаем исходные пиксели только там где маска
        cleaned = binary & mask
        cleared_.append(cleaned)
    return cleared_


"""
можно так же написать функцию которая удаляет излишни
которые находятся не в основной массе
"""

# Функция для фильтрации кластеров по площади и радиусу
def filter_clusters(cleaned_list: list, min_area: int = 3, # минимальная площадь кластера
                    min_radius: float = 6.5, # минимальный радиус кластера
                    cluster_threshold: int = 100, # если кластеров меньше этого значения, не фильтровать
                    sigma: float = 1.
                    ) -> list:
    
    filtered_list = []

    for cleaned in cleaned_list:
        # Находим связные компоненты
        labeled, num_features = label(cleaned)
        slices = find_objects(labeled)

        # Если кластеров мало — не фильтруем, возвращаем как есть
        if num_features <= cluster_threshold:
            filtered_list.append(cleaned)
            continue
        # Иначе фильтруем по площади и радиусу
        filtered = np.zeros_like(cleaned)
        # Проходим по всем найденным объектам
        for i, slc in enumerate(slices):
            if slc is None:
                continue
            # Вычисляем высоту, ширину, площадь и радиус кластера
            h = slc[0].stop - slc[0].start
            w = slc[1].stop - slc[1].start
            area = h * w
            radius = ((h**2 + w**2) ** 0.5) / 2
            # Если кластер удовлетворяет условиям по площади и радиусу, сохраняем его в итоговом изображении
            if area > min_area and radius > min_radius:
                filtered[labeled == i + 1] = True

        # морфологическое сглаживание
        blurred = gaussian_filter(filtered.astype(float), sigma=sigma)
        smoothed = blurred > 0.5

        filtered_list.append(smoothed)

    return filtered_list

