import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from glob import glob

image_dir = 'train_dataset_skoltech_train/train/images'
mask_dir = 'train_dataset_skoltech_train/train/masks'
output_report = 'data_work/complete_image_analysis_report.json'
output_graphs_dir = 'data_work/graphs'

# Создаем папку для сохранения графиков, если ее нет
os.makedirs(output_graphs_dir, exist_ok=True)

# Словарь для хранения всей информации об изображениях и масках
report_data = {}

# Функция для анализа изображения
def analyze_image(filepath):
    with rasterio.open(filepath) as dataset:
        # Информация об изображении
        width, height = dataset.width, dataset.height
        num_channels = dataset.count
        crs = str(dataset.crs)
        transform = dataset.transform
        channels_info = []

        for i in range(1, num_channels + 1):
            band = dataset.read(i)
            min_val, max_val = band.min(), band.max()
            mean_val = band.mean()
            channels_info.append({
                'channel': i,
                'min': float(min_val),
                'max': float(max_val),
                'mean': float(mean_val)
            })

            # Построение гистограммы
            plt.figure(figsize=(10, 4))
            plt.hist(band.flatten(), bins=50, alpha=0.7)
            plt.title(f"Histogram for Channel {i} - {os.path.basename(filepath)}")
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(output_graphs_dir, f"{os.path.basename(filepath).split('.')[0]}_channel_{i}_histogram.png"))
            plt.close()

        # Структура для хранения информации об изображении
        image_info = {
            'width': width,
            'height': height,
            'channels': num_channels,
            'crs': crs,
            'transform': str(transform),
            'channels_info': channels_info
        }
        return image_info

# Функция для проверки соответствия маски и изображения
def check_mask_compatibility(image_path, mask_path):
    with rasterio.open(image_path) as img, rasterio.open(mask_path) as mask:
        img_width, img_height = img.width, img.height
        mask_width, mask_height = mask.width, mask.height
        crs_match = img.crs == mask.crs
        transform_match = img.transform == mask.transform
        return {
            'dimensions_match': img_width == mask_width and img_height == mask_height,
            'crs_match': crs_match,
            'transform_match': transform_match
        }

# Функция для исправления трансформации маски, чтобы она соответствовала изображению
def fix_mask_transform(image_path, mask_path, fixed_mask_path):
    with rasterio.open(image_path) as img, rasterio.open(mask_path) as mask:
        mask_data = mask.read(1)
        profile = mask.profile
        profile.update({
            'transform': img.transform,
            'crs': img.crs
        })
        with rasterio.open(fixed_mask_path, 'w', **profile) as dst:
            dst.write(mask_data, 1)

# Анализ всех изображений и масок
image_files = sorted(glob(os.path.join(image_dir, '*.tif')))
mask_files = sorted(glob(os.path.join(mask_dir, '*.tif')))

for image_file, mask_file in zip(image_files, mask_files):
    image_name = os.path.basename(image_file)
    mask_name = os.path.basename(mask_file)
    image_info = analyze_image(image_file)
    mask_compatibility = check_mask_compatibility(image_file, mask_file)

    # Если трансформации не совпадают, исправляем маску
    if not mask_compatibility['transform_match'] or not mask_compatibility['crs_match']:
        fixed_mask_path = mask_file.replace('.tif', '_fixed.tif')
        fix_mask_transform(image_file, mask_file, fixed_mask_path)
        mask_file = fixed_mask_path  # Обновляем путь к исправленной маске
        mask_compatibility = check_mask_compatibility(image_file, mask_file)

    # Добавляем информацию в отчет
    report_data[image_name] = {
        'image_info': image_info,
        'mask_compatibility': mask_compatibility
    }

with open(output_report, 'w') as f:
    json.dump(report_data, f, indent=4)


