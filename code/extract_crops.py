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

    return ndvi, evi, ndwi, mtci, swir

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
    sentinel_file_oct = '../raw/47PQQ_2018-10-31.tif'
    sentinel_file_nov = '../raw/47PQQ_2018-11-30.tif'
    sentinel_file_dec = '../raw/47PQQ_2018-12-31.tif'
    label = rasterio.open(label_file)
    tile_oct = rasterio.open(sentinel_file_oct)
    tile_nov = rasterio.open(sentinel_file_nov)
    tile_dec = rasterio.open(sentinel_file_dec)
    tile_id = os.path.basename(sentinel_file_oct).split('_')[0]
    labels = extract_overlap(label, tile_oct, tile_id)
    ndvi_oct, evi_oct, ndwi_oct, mtci_oct, swir_oct = compute_indices(tile_oct)
    ndvi_nov, evi_nov, ndwi_nov, mtci_nov, swir_nov = compute_indices(tile_nov)
    ndvi_dec, evi_dec, ndwi_dec, mtci_dec, swir_dec = compute_indices(tile_dec)
    features = [
        ndvi_oct, evi_oct, ndwi_oct, mtci_oct, swir_oct,
        ndvi_nov, evi_nov, ndwi_nov, mtci_nov, swir_nov,
        ndvi_dec, evi_dec, ndwi_dec, mtci_dec, swir_dec
    ]
    samples_per_class = 2000
    dataset = []
    unique_classes = np.unique(labels)
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

    known_mask = np.isin(labels, list(class_map.keys()))
    others_mask = ~known_mask
    rows = sample_pixels(others_mask, features, samples_per_class)

    for row in rows:
        row.append(others)
        dataset.append(row)

    columns = [
        'ndvi_oct', 'evi_oct', 'ndwi_oct', 'mtci_oct', 'swir_oct',
        'ndvi_nov', 'evi_nov', 'ndwi_nov', 'mtci_nov', 'swir_nov',
        'ndvi_dec', 'evi_dec', 'ndwi_dec', 'mtci_dec', 'swir_dec',
        'class'
    ]
    df = pd.DataFrame(dataset, columns=columns)
    df.to_csv('../csvs/rawWithBuffer.csv', index=False)