import os
import rasterio
import numpy as np
import json
from glob import glob
import matplotlib.pyplot as plt

def tile_image(image_path, mask_path, tile_size=256, output_dir='data_work/processed_data'):
    # Загружаем изображение и маску с использованием rasterio
    with rasterio.open(image_path) as src:
        image = src.read()  # (channels, height, width)
    
    with rasterio.open(mask_path) as src_mask:
        mask = src_mask.read(1)  # Считываем только один канал для маски

    image_height, image_width = image.shape[1], image.shape[2]
    mask_height, mask_width = mask.shape

    # Создаем директории для хранения тайлов
    os.makedirs(output_dir, exist_ok=True)
    tiles_dir = os.path.join(output_dir, 'tiles')
    os.makedirs(tiles_dir, exist_ok=True)

    augmentation_report = {}

    tile_count = 0
    for i in range(0, image_height, tile_size):
        for j in range(0, image_width, tile_size):
            # Проверяем, что текущий тайл имеет ненулевые размеры
            if i + tile_size > image_height or j + tile_size > image_width:
                print(f"Пропуск тайла {i}_{j} из-за нулевого размера или выхода за границы изображения")
                continue

            tile = image[:, i:i + tile_size, j:j + tile_size]
            tile_mask = mask[i:i + tile_size, j:j + tile_size]

            # Если размеры тайлов и масок несовпадают, используем паддинг
            if tile.shape[1] != tile_size or tile.shape[2] != tile_size:
                padded_tile = np.zeros((tile.shape[0], tile_size, tile_size), dtype=tile.dtype)
                padded_tile[:, :tile.shape[1], :tile.shape[2]] = tile
                tile = padded_tile

            if tile_mask.shape[0] != tile_size or tile_mask.shape[1] != tile_size:
                padded_mask = np.zeros((tile_size, tile_size), dtype=tile_mask.dtype)
                padded_mask[:tile_mask.shape[0], :tile_mask.shape[1]] = tile_mask
                tile_mask = padded_mask

            # Сохраняем тайлы
            tile_filename = f"tile_{i}_{j}.npy"
            mask_filename = f"mask_{i}_{j}.npy"
            np.save(os.path.join(tiles_dir, tile_filename), tile)
            np.save(os.path.join(tiles_dir, mask_filename), tile_mask)

            # Записываем данные в отчет
            ndvi = (tile[3] - tile[2]) / (tile[3] + tile[2] + 1e-6)
            ndwi = (tile[1] - tile[3]) / (tile[1] + tile[3] + 1e-6)
            augmentation_report[f"{tile_count}_tile_{i}_{j}"] = {
                "tile_shape": tile.shape,
                "mask_shape": tile_mask.shape,
                "ndwi_min": float(np.min(ndwi)),
                "ndwi_max": float(np.max(ndwi)),
                "ndvi_min": float(np.min(ndvi)),
                "ndvi_max": float(np.max(ndvi))
            }

            tile_count += 1

    # Сохраняем отчет в JSON
    report_path = os.path.join(output_dir, 'augmentation_report.json')
    with open(report_path, 'w') as f:
        json.dump(augmentation_report, f, indent=4)

    print("Отчет сохранен по пути:", report_path)

def visualize_example(image_path, mask_path, output_dir='data_work/processed_data'):
    # Загрузка примера изображения и маски с использованием rasterio
    with rasterio.open(image_path) as src:
        image = src.read()
    with rasterio.open(mask_path) as src_mask:
        mask = src_mask.read(1)

    # Визуализация изображения и маски
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(np.moveaxis(image[:3], 0, -1))  # Визуализируем первые 3 канала
    ax[0].set_title('Image')
    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title('Mask')
    plt.show()

# Основной код
image_dir = 'train_dataset_skoltech_train/train/images'
mask_dir = 'train_dataset_skoltech_train/train/masks'
output_dir = 'data_work/processed_data'

# Получаем все изображения и маски
image_files = sorted(glob(os.path.join(image_dir, '*.tif')))
mask_files = sorted(glob(os.path.join(mask_dir, '*.tif')))

# Проверяем, что списки изображений и масок не пустые
if not image_files or not mask_files:
    print("Ошибка: файлы изображений или масок не найдены. Проверьте пути и наличие файлов в директориях.")
else:
    # Обрабатываем каждое изображение и маску
    for image_file, mask_file in zip(image_files, mask_files):
        tile_image(image_file, mask_file, output_dir=output_dir)

    visualize_example(image_files[0], mask_files[0])
