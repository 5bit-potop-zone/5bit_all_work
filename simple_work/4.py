import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from glob import glob

def plot_histograms(image_path, mask_path, output_dir, num_bins=50):
    """
    Функция для построения гистограмм распределения значений каждого канала изображения и маски.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Загружаем изображение и маску
    with rasterio.open(image_path) as img:
        image = img.read()  # Считываем все каналы
        num_channels = image.shape[0]
    
    with rasterio.open(mask_path) as msk:
        mask = msk.read(1)  # Считываем маску (1 канал)
    
    for channel in range(num_channels):
        plt.figure(figsize=(8, 6))
        plt.hist(image[channel].ravel(), bins=num_bins, color='blue', alpha=0.7)
        plt.title(f'Гистограмма распределения значений для Канала {channel + 1}')
        plt.xlabel("Значения")
        plt.ylabel("Частота")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"histogram_channel_{channel + 1}.png"))
        plt.close()

    plt.figure(figsize=(8, 6))
    plt.hist(mask.ravel(), bins=num_bins, color='green', alpha=0.7)
    plt.title('Гистограмма распределения значений для Маски')
    plt.xlabel("Значения")
    plt.ylabel("Частота")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "histogram_mask.png"))
    plt.close()

    print(f"Гистограммы сохранены в {output_dir}")

image_dir = "data_work/data_work/train_dataset_skoltech_train/train/images"
mask_dir = "data_work/data_work/train_dataset_skoltech_train/train/masks"
output_dir = "data_work/data_work/simple_work/histograms"

image_files = sorted(glob(os.path.join(image_dir, "*.tif")))
mask_files = sorted(glob(os.path.join(mask_dir, "*.tif")))

for image_file, mask_file in zip(image_files, mask_files):
    image_name = os.path.splitext(os.path.basename(image_file))[0]
    mask_name = os.path.splitext(os.path.basename(mask_file))[0]
    specific_output_dir = os.path.join(output_dir, f"{image_name}_{mask_name}")
    
    plot_histograms(image_file, mask_file, specific_output_dir)
