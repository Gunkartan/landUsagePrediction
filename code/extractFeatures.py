import os.path
import rasterio
import numpy as np
import pandas as pd
from labelExtractor import extract_overlap

def extractRawVals(alignedOverlap: np.ndarray, ndvi: np.ndarray, ndwi: np.ndarray, evi: np.ndarray, ndbi: np.ndarray):
    rows, cols = np.where(alignedOverlap != 0)
    features = []

    for r, c in zip(rows, cols):
        vals = [ndvi[r, c], ndwi[r, c], evi[r, c], ndbi[r, c]]
        label = 1 if str(alignedOverlap[r, c]).startswith('4') else 0
        features.append(vals + [label])

    return features

def createCSV(features: list[list[any]], columns: list[str]):
    df = pd.DataFrame(features, columns=columns)
    df = df.round(3)
    df.to_csv('raw.csv', index=False)

if __name__ == '__main__':
    labelFile = f'../rasterized/2018.tif'
    sentinelFile = f'../raw/47PQQ_2018-10-31.tif'
    label = rasterio.open(labelFile)
    tile = rasterio.open(sentinelFile)
    tileID = os.path.basename(sentinelFile).split('_')[0][0:]
    alignedOverlap = extract_overlap(label, tile, tileID)
    columns = ['NDVI', 'NDWI', 'EVI', 'NDBI', 'Water']
    blue = tile.read(1).astype('float32')
    green = tile.read(2).astype('float32')
    red = tile.read(3).astype('float32')
    nir = tile.read(7).astype('float32')
    swir = tile.read(8).astype('float32')
    ndvi = (nir - red) / (nir + red)
    ndwi = (green - nir) / (green + nir)
    evi = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))
    ndbi = (swir - nir) / (swir + nir)
    features = extractRawVals(alignedOverlap, ndvi, ndwi, evi, ndbi)
    createCSV(features, columns)