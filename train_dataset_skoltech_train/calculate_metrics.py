import os
import rasterio
import argparse
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

from shapely import affinity
from shapely import Point

from sklearn.metrics import f1_score


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--osm_path", type=str, help="Path to OSM file", required=True) 
    arg("--masks_path", type=str, help="Path to masks", required=True)
    arg("--preds_path", type=str, help="Path to preds", required=True)
    arg("--pre_gt_path", type=str, help="Path to pre gt", required=True)
    arg("--post_gt_path", type=str, help="Path to post gt", required=True)
    arg("--pre_pred_path", type=str, help="Path to pre pred", required=True)
    arg("--post_pred_path", type=str, help="Path to post pred", required=True)
    return parser.parse_args()


def flooded_houses(
    osm_path: str, 
    lats: np.ndarray, 
    lons: np.ndarray, 
    pred: np.ndarray, 
    ground_truth: np.ndarray 
):
    gdf = gpd.read_file(osm_path)
    gdf = gdf.to_crs(4326)
    gdf.tags.unique()

    flooded_pred = []
    flooded_gt = []
    pred = pred.flatten()  # Flatten the prediction array
    ground_truth = ground_truth.flatten()  # Flatten the ground_truth array
    
    for _, row in gdf.iterrows():
        polygon = row.geometry
        # Scale the polygon for more accurate coverage
        scaled_polygon = affinity.scale(polygon, xfact=1.5, yfact=1.5)
        
        # Get the polygon's bounding box (xmin, ymin, xmax, ymax)
        xmin, ymin, xmax, ymax = scaled_polygon.bounds

        # Find the indices of points that fall inside the bounding box of the polygon
        selected_indices = np.where((ymin <= lats) & (lats <= ymax) & (xmin <= lons) & (lons <= xmax))
        
        lats_to_check = lats[selected_indices]
        lons_to_check = lons[selected_indices]
        flood_pred_to_check = pred[selected_indices]
        flood_gt_to_check = ground_truth[selected_indices]

        # Check if at least one point inside the polygon is flooded in the prediction mask
        is_flooded_pred = any(
            flood_pred_to_check[i] and scaled_polygon.contains(Point(lons_to_check[i], lats_to_check[i]))
            for i in range(len(flood_pred_to_check))
        )

        # Check if at least one point inside the polygon is flooded in the ground truth mask
        is_flooded_gt = any(
            flood_gt_to_check[i] and scaled_polygon.contains(Point(lons_to_check[i], lats_to_check[i]))
            for i in range(len(flood_gt_to_check))
        )

        flooded_pred.append(1 if is_flooded_pred else 0)
        flooded_gt.append(1 if is_flooded_gt else 0)

    return f1_score(flooded_gt, flooded_pred, average='macro')


def load_raster(file_path):
    """Load a raster image and flatten it into a 1D array."""
    with rasterio.open(file_path) as src:
        data = src.read(1)  # Read the first band
    return data.flatten()


def calculate_f1_score(dir1, dir2):
    """Calculate the F1 score between corresponding images in two directories."""
    f1_scores = []
    
    # List all files in the directories
    files1 = sorted(os.listdir(dir1))
    files2 = sorted(os.listdir(dir2))
    
    
    for file1, file2 in zip(files1, files2):

        file_path1 = os.path.join(dir1, file1)
        file_path2 = os.path.join(dir2, file2)
        
        # Load the images
        img1 = load_raster(file_path1)
        img2 = load_raster(file_path2)
        
        # Calculate F1 score
        f1 = f1_score(img1, img2, average='macro')
        f1_scores.append(f1)
    
    # Calculate average F1 score across all image pairs
    average_f1 = np.mean(f1_scores)
    return average_f1


def main():

    args = get_args()

    masks_path = args.masks_path
    preds_path = args.preds_path

    f1_water = calculate_f1_score(masks_path, preds_path)

    osm_path = args.osm_path

    pre_gt_path = args.pre_gt_path
    post_gt_path = args.post_gt_path

    pre_pred_path = args.pre_pred_path
    post_pred_path = args.post_pred_path

    with rasterio.open(pre_gt_path) as multi_band_src:
        pre_mask = multi_band_src.read(1)
        pre_height, pre_width = pre_mask.shape
        pre_cols, pre_rows = np.meshgrid(np.arange(pre_width), np.arange(pre_height))
        pre_x, pre_y = rasterio.transform.xy(multi_band_src.transform, pre_rows, pre_cols) 
        pre_lons, pre_lats = np.array(pre_x), np.array(pre_y)

    with rasterio.open(pre_pred_path) as multi_band_src:
        pre_pred = multi_band_src.read(1)

    with rasterio.open(post_gt_path) as multi_band_src:
        post_mask = multi_band_src.read(1)
        post_height, post_width = post_mask.shape
        post_cols, post_rows = np.meshgrid(np.arange(post_width), np.arange(post_height))
        post_x, post_y = rasterio.transform.xy(multi_band_src.transform, post_rows, post_cols) 
        post_lons, post_lats = np.array(post_x), np.array(post_y)

    with rasterio.open(post_pred_path) as multi_band_src:
        post_pred = multi_band_src.read(1)

    pre_f1 = flooded_houses(osm_path, pre_lats, pre_lons, pre_pred, pre_mask)
    post_f1 = flooded_houses(osm_path, post_lats, post_lons, post_pred, post_mask)
    avg_f1_business = (pre_f1 + post_f1) / 2

    print(f"F1-Score: {(f1_water + avg_f1_business) / 2 :.3f}")

if __name__ == "__main__":
    main()
