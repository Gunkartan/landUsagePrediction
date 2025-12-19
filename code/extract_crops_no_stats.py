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
from scipy.ndimage import binary_erosion

def apply_buffer(label_raster, pixel_size=10):
    buffer_pixel = int(30 / pixel_size)
    mask = label_raster != 0
    structure = np.ones((3, 3))
    eroded = binary_erosion(mask, structure=structure, iterations=buffer_pixel)

    return np.where(eroded, label_raster, 0)

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
    sentinel_file_oct = '../raw/47PQQ_2018-10-31.tif'
    sentinel_file_nov = '../raw/47PQQ_2018-11-30.tif'
    sentinel_file_dec = '../raw/47PQQ_2018-12-31.tif'

    label = rasterio.open(label_file)
    tile_oct = rasterio.open(sentinel_file_oct)
    tile_nov = rasterio.open(sentinel_file_nov)
    tile_dec = rasterio.open(sentinel_file_dec)
    tile_id = os.path.basename(sentinel_file_oct).split('_')[0]
    aligned_overlap = apply_buffer(extract_overlap(label, tile_oct, tile_id))
    aligned_overlap = extract_overlap(label, tile_oct, tile_id)

    blue_oct = tile_oct.read(1).astype('float32')
    green_oct = tile_oct.read(2).astype('float32')
    red_oct = tile_oct.read(3).astype('float32')
    re_early_oct = tile_oct.read(4).astype('float32')
    re_mid_oct = tile_oct.read(5).astype('float32')
    nir_oct = tile_oct.read(7).astype('float32')
    swir_oct = tile_oct.read(8).astype('float32')
    swir_long_oct = tile_oct.read(9).astype('float32')
    nir_narrow_oct = tile_oct.read(10).astype('float32')

    ndvi_oct = (nir_oct - red_oct) / (nir_oct + red_oct)
    ndwi_oct = (green_oct - nir_oct) / (green_oct + nir_oct)
    evi_oct = 2.5 * (nir_oct - red_oct) / (nir_oct + 6 * red_oct - 7.5 * blue_oct + 1)
    ndbi_oct = (swir_oct - nir_oct) / (swir_oct + nir_oct)
    mtci_oct = (nir_oct - re_mid_oct) / (re_mid_oct - red_oct)

    blue_nov = tile_nov.read(1).astype('float32')
    green_nov = tile_nov.read(2).astype('float32')
    red_nov = tile_nov.read(3).astype('float32')
    re_early_nov = tile_nov.read(4).astype('float32')
    re_mid_nov = tile_nov.read(5).astype('float32')
    nir_nov = tile_nov.read(7).astype('float32')
    swir_nov = tile_nov.read(8).astype('float32')
    swir_long_nov = tile_nov.read(9).astype('float32')
    nir_narrow_nov = tile_nov.read(10).astype('float32')

    ndvi_nov = (nir_nov - red_nov) / (nir_nov + red_nov)
    ndwi_nov = (green_nov - nir_nov) / (green_nov + nir_nov)
    evi_nov = 2.5 * (nir_nov - red_nov) / (nir_nov + 6 * red_nov - 7.5 * blue_nov + 1)
    ndbi_nov = (swir_nov - nir_nov) / (swir_nov + nir_nov)
    mtci_nov = (nir_nov - re_mid_nov) / (re_mid_nov - red_nov)

    blue_dec = tile_dec.read(1).astype('float32')
    green_dec = tile_dec.read(2).astype('float32')
    red_dec = tile_dec.read(3).astype('float32')
    re_early_dec = tile_dec.read(4).astype('float32')
    re_mid_dec = tile_dec.read(5).astype('float32')
    nir_dec = tile_dec.read(7).astype('float32')
    swir_dec = tile_dec.read(8).astype('float32')
    swir_long_dec = tile_dec.read(9).astype('float32')
    nir_narrow_dec = tile_dec.read(10).astype('float32')

    ndvi_dec = (nir_dec - red_dec) / (nir_dec + red_dec)
    ndwi_dec = (green_dec - nir_dec) / (green_dec + nir_dec)
    evi_dec = 2.5 * (nir_dec - red_dec) / (nir_dec + 6 * red_dec - 7.5 * blue_dec + 1)
    ndbi_dec = (swir_dec - nir_dec) / (swir_dec + nir_dec)
    mtci_dec = (nir_dec - re_mid_dec) / (re_mid_dec - red_dec)

    indices_dict = {
        'ndvi_oct': ndvi_oct,
        'ndwi_oct': ndwi_oct,
        'evi_oct': evi_oct,
        'ndbi_oct': ndbi_oct,
        'mtci_oct': mtci_oct,
        'ndvi_nov': ndvi_nov,
        'ndwi_nov': ndwi_nov,
        'evi_nov': evi_nov,
        'ndbi_nov': ndbi_nov,
        'mtci_nov': mtci_nov,
        'ndvi_dec': ndvi_dec,
        'ndwi_dec': ndwi_dec,
        'evi_dec': evi_dec,
        'ndbi_dec': ndbi_dec,
        'mtci_dec': mtci_dec
    }

    block_size = 1024
    num_rows, num_cols = aligned_overlap.shape
    cols = ['Label'] + list(indices_dict.keys()) + ['Crops']

    for row_start in range(0, num_rows, block_size):
        for col_start in range(0, num_cols, block_size):
            row_end = min(row_start + block_size, num_rows)
            col_end = min(col_start + block_size, num_cols)
            aligned_block = aligned_overlap[row_start:row_end, col_start:col_end]
            indices_block = {name: arr[row_start:row_end, col_start:col_end] for name, arr in indices_dict.items()}
            features_block = extract_raw_vals_fast(aligned_block, indices_block)
            create_csv(features_block, cols, row_start == 0 and col_start == 0)