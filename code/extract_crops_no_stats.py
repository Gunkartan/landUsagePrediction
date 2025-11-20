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
    sentinel_file_oct = '../raw/47PQQ_2018-10-31.tif'
    sentinel_file_nov = '../raw/47PQQ_2018-11-30.tif'
    sentinel_file_dec = '../raw/47PQQ_2018-12-31.tif'

    label = rasterio.open(label_file)
    tile_oct = rasterio.open(sentinel_file_oct)
    tile_nov = rasterio.open(sentinel_file_nov)
    tile_dec = rasterio.open(sentinel_file_dec)
    tile_id = os.path.basename(sentinel_file_oct).split('_')[0]
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
    bsi_oct = ((swir_oct + red_oct) - (nir_oct + blue_oct)) / ((swir_oct + red_oct) + (nir_oct + blue_oct))
    savi_oct = (1.5 * (nir_oct - red_oct)) / (nir_oct + red_oct + 0.5)
    ndsmi_oct = (nir_oct - swir_oct) / (nir_oct + swir_oct)
    nmdi_oct = (nir_oct - (swir_oct - swir_long_oct)) / (nir_oct + (swir_oct - swir_long_oct))
    mndwi_oct = (green_oct - swir_oct) / (green_oct + swir_oct)
    ndmi_oct = (nir_oct - swir_oct) / (nir_oct + swir_oct)
    ndii_oct = (nir_oct - swir_long_oct) / (nir_oct + swir_long_oct)

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
    bsi_nov = ((swir_nov + red_nov) - (nir_nov + blue_nov)) / ((swir_nov + red_nov) + (nir_nov + blue_nov))
    savi_nov = (1.5 * (nir_nov - red_nov)) / (nir_nov + red_nov + 0.5)
    ndsmi_nov = (nir_nov - swir_nov) / (nir_nov + swir_nov)
    nmdi_nov = (nir_nov - (swir_nov - swir_long_nov)) / (nir_nov + (swir_nov - swir_long_nov))
    mndwi_nov = (green_nov - swir_nov) / (green_nov + swir_nov)
    ndmi_nov = (nir_nov - swir_nov) / (nir_nov + swir_nov)
    ndii_nov = (nir_nov - swir_long_nov) / (nir_nov + swir_long_nov)

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
    bsi_dec = ((swir_dec + red_dec) - (nir_dec + blue_dec)) / ((swir_dec + red_dec) + (nir_dec + blue_dec))
    savi_dec = (1.5 * (nir_dec - red_dec)) / (nir_dec + red_dec + 0.5)
    ndsmi_dec = (nir_dec - swir_dec) / (nir_dec + swir_dec)
    nmdi_dec = (nir_dec - (swir_dec - swir_long_dec)) / (nir_dec + (swir_dec - swir_long_dec))
    mndwi_dec = (green_dec - swir_dec) / (green_dec + swir_dec)
    ndmi_dec = (nir_dec - swir_dec) / (nir_dec + swir_dec)
    ndii_dec = (nir_dec - swir_long_dec) / (nir_dec + swir_long_dec)

    indices_dict = {
        'ndvi_oct': ndvi_oct,
        'ndwi_oct': ndwi_oct,
        'evi_oct': evi_oct,
        'ndbi_oct': ndbi_oct,
        'bsi_oct': bsi_oct,
        'savi_oct': savi_oct,
        'ndsmi_oct': ndsmi_oct,
        'nmdi_oct': nmdi_oct,
        'mndwi_oct': mndwi_oct,
        'ndmi_oct': ndmi_oct,
        'ndii_oct': ndii_oct,
        'ndvi_nov': ndvi_nov,
        'ndwi_nov': ndwi_nov,
        'evi_nov': evi_nov,
        'ndbi_nov': ndbi_nov,
        'bsi_nov': bsi_nov,
        'savi_nov': savi_nov,
        'ndsmi_nov': ndsmi_nov,
        'nmdi_nov': nmdi_nov,
        'mndwi_nov': mndwi_nov,
        'ndmi_nov': ndmi_nov,
        'ndii_nov': ndii_nov,
        'ndvi_dec': ndvi_dec,
        'ndwi_dec': ndwi_dec,
        'evi_dec': evi_dec,
        'ndbi_dec': ndbi_dec,
        'bsi_dec': bsi_dec,
        'savi_dec': savi_dec,
        'ndsmi_dec': ndsmi_dec,
        'nmdi_dec': nmdi_dec,
        'mndwi_dec': mndwi_dec,
        'ndmi_dec': ndmi_dec,
        'ndii_dec': ndii_dec
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