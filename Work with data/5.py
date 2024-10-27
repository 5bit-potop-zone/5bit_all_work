import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from glob import glob

def plot_channel_histograms(files, channel_index, channel_name, num_bins=50):
    """
    Функция для построения гистограммы значений одного канала для нескольких файлов.
    
    files: список путей к файлам изображений
    channel_index: индекс канала для анализа (начиная с 1)
    channel_name: название канала для отображения в заголовке
    num_bins: количество бинов для гистограммы
    """
    plt.figure(figsize=(14, 8))  # Увеличиваем размер графика
    
    for file in files:
        with rasterio.open(file) as img:
            # Чтение одного канала
            channel_data = img.read(channel_index).ravel()
            plt.hist(channel_data, bins=num_bins, alpha=0.6, label=os.path.basename(file),
                     histtype='step', linewidth=2.0)  # Увеличиваем толщину линий
    
    plt.title(f'Гистограмма значений для канала {channel_name} (Канал {channel_index})', fontsize=16)
    plt.xlabel("Значения", fontsize=14)
    plt.ylabel("Частота", fontsize=14)
    plt.legend(loc="upper right", fontsize="small")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Добавляем сетку для удобства
    plt.show()

# Путь к папке с изображениями
image_dir = "data_work/data_work/train_dataset_skoltech_train/train/images"
image_files = sorted(glob(os.path.join(image_dir, "*.tif")))

# Выбираем только 7 первых файлов для анализа
selected_files = image_files[:7]

# Построение гистограмм для каждого канала
channel_names = ["B02 - Blue", "B03 - Green", "B04 - Red", "B05 - Veg Red Edge", 
                 "B06 - Veg Red Edge", "B07 - Veg Red Edge", "B08 - NIR", 
                 "B8A - Narrow NIR", "B11 - SWIR", "B12 - SWIR"]

for channel_index, channel_name in enumerate(channel_names, start=1):
    plot_channel_histograms(selected_files, channel_index, channel_name)
