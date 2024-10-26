import os
import numpy as np
import pandas as pd
import rasterio
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def pad_tile(tile, tile_size):
    if tile.ndim == 2:  # For masks (single-channel)
        padded_tile = np.zeros((tile_size, tile_size), dtype=tile.dtype)
        padded_tile[:tile.shape[0], :tile.shape[1]] = tile
    else:  
        padded_tile = np.zeros((tile_size, tile_size, tile.shape[2]), dtype=tile.dtype)
        padded_tile[:tile.shape[0], :tile.shape[1], :] = tile
    return padded_tile

def process_tile(image, mask, i, j, tile_size):
    tile = image[i:i + tile_size, j:j + tile_size]
    tile_mask = mask[i:i + tile_size, j:j + tile_size]

    if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
        tile = pad_tile(tile, tile_size)
        tile_mask = pad_tile(tile_mask, tile_size)
    
    water_label = 1 if np.any(tile_mask > 0) else 0
    water_coverage = np.sum(tile_mask > 0) / (tile_size * tile_size)

    return tile, tile_mask, water_label, water_coverage

def save_tile(tile, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=tile.shape[0],
        width=tile.shape[1],
        count=tile.shape[2] if tile.ndim == 3 else 1,
        dtype=tile.dtype
    ) as dst:
        if tile.ndim == 3:
            for k in range(tile.shape[2]):
                dst.write(tile[:, :, k], k + 1)
        else:
            dst.write(tile, 1)

def save_mask(mask, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if mask.size == 0:
        print(f"Warning: Mask with zero dimensions, creating empty mask at {output_path}")
        mask = np.zeros((256, 256), dtype=mask.dtype)  # Replace with empty mask if zero dimensions
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=mask.shape[0],
        width=mask.shape[1],
        count=1,
        dtype=mask.dtype
    ) as dst:
        dst.write(mask, 1)

def tile_image_and_mask(image_path, mask_path, tile_size, output_dir, csv_data):
    try:
        with rasterio.open(image_path) as src_img:
            image = src_img.read().transpose(1, 2, 0)
        with rasterio.open(mask_path) as src_mask:
            mask = src_mask.read(1)
    except Exception as e:
        print(f"Error loading image or mask: {e}")
        return

    image_height, image_width = image.shape[:2]
    for i in range(0, image_height, tile_size):
        for j in range(0, image_width, tile_size):
            tile, tile_mask, water_label, water_coverage = process_tile(image, mask, i, j, tile_size)
            
            tile_filename = f"tile_{os.path.splitext(os.path.basename(image_path))[0]}_{i}_{j}.tif"
            mask_filename = f"mask_{os.path.splitext(os.path.basename(mask_path))[0]}_{i}_{j}.tif"
            tile_output_path = os.path.join(output_dir, "images", tile_filename)
            mask_output_path = os.path.join(output_dir, "masks", mask_filename)
            
            try:
                save_tile(tile, tile_output_path)
                save_mask(tile_mask, mask_output_path)
            except Exception as e:
                print(f"Error saving tile or mask at {tile_output_path} or {mask_output_path}: {e}")
                continue
            
            csv_data.append({
                "image_path": tile_output_path,
                "mask_path": mask_output_path,
                "original_image": os.path.basename(image_path),
                "tile_x": i,
                "tile_y": j,
                "water_label": water_label,
                "water_coverage": water_coverage
            })

def create_dataset(image_dir, mask_dir, output_dir, tile_size=256):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    
    csv_data = []
    image_files = sorted(glob(os.path.join(image_dir, "*.tif")))
    mask_files = sorted(glob(os.path.join(mask_dir, "*.tif")))

    if len(image_files) != len(mask_files):
        print("Warning: The number of images and masks does not match.")
    
    for image_file, mask_file in tqdm(zip(image_files, mask_files), total=len(image_files), desc="Processing tiles"):
        tile_image_and_mask(image_file, mask_file, tile_size, output_dir, csv_data)
    
    df = pd.DataFrame(csv_data)
    csv_path = os.path.join(output_dir, "dataset.csv")
    df.to_csv(csv_path, index=False)
    print(f"CSV dataset created at {csv_path}")

image_dir = "train_dataset_skoltech_train/train/images"
mask_dir = "train_dataset_skoltech_train/train/masks"
output_dir = "data_work_ds_and_cut/processed_data"
tile_size = 256

create_dataset(image_dir, mask_dir, output_dir, tile_size)
