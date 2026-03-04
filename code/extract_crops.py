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
#Reservoir: 4201
import os.path
import rasterio
import numpy as np
import pandas as pd
from labelExtractor import extract_overlap
from scipy.ndimage import binary_erosion

def compute_indices(tile):
    blue = tile.read(1).astype(float)
    green = tile.read(2).astype(float)
    red = tile.read(3).astype(float)
    re_early = tile.read(4).astype(float)
    re_mid = tile.read(5).astype(float)
    nir = tile.read(7).astype(float)
    narrow_nir = tile.read(8).astype(float)
    swir = tile.read(9).astype(float)
    ndvi = (nir - red) / (nir + red)
    evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
    ndwi = (green - narrow_nir) / (green + narrow_nir)
    mtci = (re_mid - re_early) / (re_early - red)
    savi = ((nir - red) / (nir + red + 0.5)) * 1.5
    ndti = (swir - red) / (swir + red)
    bsi = ((swir + red) - (nir + blue)) / ((swir + red) + (nir + blue))

    return ndvi, evi, ndwi, mtci, swir, savi, ndti, bsi

def sample_pixels(mask, features, sample_size, buffer_pixels = 3):
    mask = binary_erosion(mask, iterations=buffer_pixels)
    idx = np.column_stack(np.where(mask))

    if len(idx) > sample_size:
        idx = idx[np.random.choice(len(idx), sample_size, replace=False)]

    rows = []

    for r, c in idx:
        row = []

        for feature in features:
            row.append(feature[r, c])

        rows.append(row)

    return rows

if __name__ == '__main__':
    label_file = '../rasterized/2018.tif'
    label_file_2020 = '../rasterized/2020.tif'
    sentinel_file_oct = '../raw/47PQQ_2018-10-31.tif'
    sentinel_file_nov = '../raw/47PQQ_2018-11-30.tif'
    sentinel_file_dec = '../raw/47PQQ_2018-12-31.tif'
    sentinel_file_oct_2020 = '../raw/47PQQ_2020-10-31.tif'
    sentinel_file_nov_2020 = '../raw/47PQQ_2020-11-30.tif'
    sentinel_file_dec_2020 = '../raw/47PQQ_2020-12-31.tif'
    label = rasterio.open(label_file)
    label_2020 = rasterio.open(label_file_2020)
    tile_oct = rasterio.open(sentinel_file_oct)
    tile_nov = rasterio.open(sentinel_file_nov)
    tile_dec = rasterio.open(sentinel_file_dec)
    tile_oct_2020 = rasterio.open(sentinel_file_oct_2020)
    tile_nov_2020 = rasterio.open(sentinel_file_nov_2020)
    tile_dec_2020 = rasterio.open(sentinel_file_dec_2020)
    tile_id = os.path.basename(sentinel_file_oct).split('_')[0]
    labels = extract_overlap(label, tile_oct, tile_id)
    labels_2020 = extract_overlap(label_2020, tile_oct_2020, tile_id)
    ndvi_oct, evi_oct, ndwi_oct, mtci_oct, swir_oct, savi_oct, ndti_oct, bsi_oct = compute_indices(tile_oct)
    ndvi_nov, evi_nov, ndwi_nov, mtci_nov, swir_nov, savi_nov, ndti_nov, bsi_nov = compute_indices(tile_nov)
    ndvi_dec, evi_dec, ndwi_dec, mtci_dec, swir_dec, savi_dec, ndti_dec, bsi_dec = compute_indices(tile_dec)
    ndvi_oct_2020, evi_oct_2020, ndwi_oct_2020, mtci_oct_2020, swir_oct_2020, savi_oct_2020, ndti_oct_2020, bsi_oct_2020 = compute_indices(tile_oct_2020)
    ndvi_nov_2020, evi_nov_2020, ndwi_nov_2020, mtci_nov_2020, swir_nov_2020, savi_nov_2020, ndti_nov_2020, bsi_nov_2020 = compute_indices(tile_nov_2020)
    ndvi_dec_2020, evi_dec_2020, ndwi_dec_2020, mtci_dec_2020, swir_dec_2020, savi_dec_2020, ndti_dec_2020, bsi_dec_2020 = compute_indices(tile_dec_2020)
    features = [
        ndvi_oct, evi_oct, ndwi_oct, mtci_oct, swir_oct, savi_oct, ndti_oct, bsi_oct,
        ndvi_nov, evi_nov, ndwi_nov, mtci_nov, swir_nov, savi_nov, ndti_nov, bsi_nov,
        ndvi_dec, evi_dec, ndwi_dec, mtci_dec, swir_dec, savi_dec, ndti_dec, bsi_dec
    ]
    features_2020 = [
        ndvi_oct_2020, evi_oct_2020, ndwi_oct_2020, mtci_oct_2020, swir_oct_2020, savi_oct_2020, ndti_oct_2020, bsi_oct_2020,
        ndvi_nov_2020, evi_nov_2020, ndwi_nov_2020, mtci_nov_2020, swir_nov_2020, savi_nov_2020, ndti_nov_2020, bsi_nov_2020,
        ndvi_dec_2020, evi_dec_2020, ndwi_dec_2020, mtci_dec_2020, swir_dec_2020, savi_dec_2020, ndti_dec_2020, bsi_dec_2020
    ]
    samples_per_class = 200000
    small_classes = [2404, 2405, 2413, 2416, 2419, 2420]
    dataset = []
    class_map = {
        2101: 'Rice',
        2204: 'Cassava',
        2205: 'Pineapple',
        2302: 'Rubber',
        2303: 'Oil palm',
        2403: 'Durian',
        2404: 'Rambutan',
        2405: 'Coconut',
        2407: 'Mango',
        2413: 'Longan',
        2416: 'Jackfruit',
        2419: 'Mangosteen',
        2420: 'Longkong',
        4201: 'Reservoir'
    }
    others = 9999

    for class_id in class_map:
        mask = labels == class_id
        rows = sample_pixels(mask, features, samples_per_class)

        for row in rows:
            row.append(class_id)
            dataset.append(row)

    for class_id in small_classes:
        mask = labels_2020 == class_id
        rows = sample_pixels(mask, features_2020, samples_per_class)

        for row in rows:
            row.append(class_id)
            dataset.append(row)

    known_mask = np.isin(labels, list(class_map.keys()))
    others_mask = ~known_mask
    rows = sample_pixels(others_mask, features, samples_per_class)

    for row in rows:
        row.append(others)
        dataset.append(row)

    known_mask_2020 = np.isin(labels_2020, list(class_map.keys()))
    others_mask_2020 = ~known_mask_2020
    rows = sample_pixels(others_mask_2020, features_2020, samples_per_class)

    for row in rows:
        row.append(others)
        dataset.append(row)

    columns = [
        'ndvi_oct', 'evi_oct', 'ndwi_oct', 'mtci_oct', 'swir_oct', 'savi_oct', 'ndti_oct', 'bsi_oct',
        'ndvi_nov', 'evi_nov', 'ndwi_nov', 'mtci_nov', 'swir_nov', 'savi_nov', 'ndti_nov', 'bsi_nov',
        'ndvi_dec', 'evi_dec', 'ndwi_dec', 'mtci_dec', 'swir_dec', 'savi_dec', 'ndti_dec', 'bsi_dec',
        'class'
    ]
    df = pd.DataFrame(dataset, columns=columns)
    df.to_csv('../csvs/rawWith2020.csv', index=False)