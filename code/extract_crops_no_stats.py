#Rice: 2101
#Cassava: 2204
#Pineapple: 2205
#Rubber: 2302
#Oil palm: 2303
#Durian: 2403
#Rambutan: 2404
#Coconut: 2405
#Mango: 2407
#Longan: 2413
#Jackfruit: 2416
#Mangosteen: 2419
#Longkong: 2420
import os.path
import rasterio
import numpy as np
import pandas as pd
from labelExtractor import extract_overlap

def extract_raw_vals(aligned_overlap: np.ndarray, indices_dict: dict):
    crop_classes = {
        2101: 0,
        2204: 1,
        2205: 2,
        2302: 3,
        2303: 4,
        2403: 5,
        2404: 6,
        2405: 7,
        2407: 8,
        2413: 9,
        2416: 10,
        2419: 11,
        2420: 12
    }
    mask = aligned_overlap != 0
    raw_vals = aligned_overlap[mask].astype(int)
    valid_mask = ~((raw_vals >= 1000) & (raw_vals < 2000) | (raw_vals >= 4000) & (raw_vals < 5000))
    raw_vals = raw_vals[valid_mask]
    combined_mask = np.zeros_like(aligned_overlap, dtype=bool)
    combined_mask[mask] = valid_mask
    labels = np.array([crop_classes.get(val, 13) for val in raw_vals], dtype=int)
    features = [raw_vals]

    for name, arr in indices_dict.items():
        features.append(arr[combined_mask])

    features = np.column_stack(features + [labels])

    return features

def extract_raw_vals_fast(aligned_overlap: np.ndarray, indices_dict: dict):
    crop_classes = {
        2101: 0,
        2204: 1,
        2205: 2,
        2302: 3,
        2303: 4,
        2403: 5,
        2404: 6,
        2405: 7,
        2407: 8,
        2413: 9,
        2416: 10,
        2419: 11,
        2420: 12
    }
    combined_mask = (aligned_overlap != 0) & ~((aligned_overlap >= 1000) & (aligned_overlap < 2000) | (aligned_overlap >= 4000) & (aligned_overlap < 5000))
    raw_vals = aligned_overlap[combined_mask]
    labels = np.array([crop_classes.get(val, 13) for val in raw_vals], dtype=int)
    features = [raw_vals]

    for name, arr in indices_dict.items():
        features.append(arr[combined_mask])

    features = np.column_stack(features + [labels])

    return features

def create_csv(features: list[list[any]], cols: list[str], first_write: bool):
    df = pd.DataFrame(features, columns=cols)

    if first_write:
        df.to_csv('../csvs/rawCrop.csv', index=False, float_format='%.3f')

    else:
        df.to_csv('../csvs/rawCrop.csv', mode='a', header=False, index=False, float_format='%.3f')

if __name__ == '__main__':
    label_file = '../rasterized/2018.tif'
    sentinel_file = '../raw/47PQQ_2018-10-31.tif'
    label = rasterio.open(label_file)
    tile = rasterio.open(sentinel_file)
    tile_id = os.path.basename(sentinel_file).split('_')[0]
    aligned_overlap = extract_overlap(label, tile, tile_id)
    blue = tile.read(1).astype('float32')
    green = tile.read(2).astype('float32')
    red = tile.read(3).astype('float32')
    nir = tile.read(7).astype('float32')
    swir = tile.read(8).astype('float32')
    swir_long = tile.read(9).astype('float32')
    ndvi = (nir - red) / (nir + red)
    ndwi = (green - nir) / (green + nir)
    evi = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))
    ndbi = (swir - nir) / (swir + nir)
    indices_dict = {
        'NDVI': ndvi,
        'NDWI': ndwi,
        'EVI': evi,
        'NDBI': ndbi
    }
    block_size = 1024
    num_rows, num_cols = aligned_overlap.shape
    cols = ['Label'] + list(indices_dict.keys()) + ['Crops']
    all_features = []

    for row_start in range(0, num_rows, block_size):
        for col_start in range(0, num_cols, block_size):
            row_end = min(row_start + block_size, num_rows)
            col_end = min(col_start + block_size, num_cols)
            aligned_block = aligned_overlap[row_start:row_end, col_start:col_end]
            indices_block = {name: arr[row_start:row_end, col_start:col_end] for name, arr in indices_dict.items()}
            features_block = extract_raw_vals(aligned_block, indices_block)
            create_csv(features_block, cols, row_start == 0 and col_start == 0)