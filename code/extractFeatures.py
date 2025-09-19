import os.path
import rasterio
import numpy as np
import pandas as pd
from labelExtractor import extract_overlap
from scipy.ndimage import uniform_filter, generic_filter

def varFilter(arr: np.ndarray):
    return np.var(arr, ddof=0)

def computeNeighborhoodFeatures(arr: np.ndarray, size: int):
    mean = uniform_filter(arr, size=size, mode='reflect')
    var = generic_filter(arr, varFilter, size=size, mode='reflect')

    return mean, var

def extractRawVals(alignedOverlap: np.ndarray, indicesDict: dict, size: int):
    rows, cols = np.where(alignedOverlap != 0)
    features = []
    stats = {}

    for name, arr in indicesDict.items():
        mean, var = computeNeighborhoodFeatures(arr, size)
        stats[name] = (arr, mean, var)

    for r, c in zip(rows, cols):
        vals = []

        for name in indicesDict.keys():
            arr, mean, var = stats[name]
            vals.extend([arr[r, c], mean[r, c], var[r, c]])

        raw = int(alignedOverlap[r, c])
        label = 1 if str(raw).startswith('4') else 0
        features.append([raw] + vals + [label])

    return features

def createCSV(features: list[list[any]], columns: list[str], firstWrite: bool):
    df = pd.DataFrame(features, columns=columns)
    df = df.round(3)

    if firstWrite:
        df.to_csv('raw.csv', index=False)

    else:
        df.to_csv('raw.csv', mode='a', header=False, index=False)

if __name__ == '__main__':
    labelFile = f'../rasterized/2018.tif'
    sentinelFile = f'../raw/47PQQ_2018-10-31.tif'
    label = rasterio.open(labelFile)
    tile = rasterio.open(sentinelFile)
    tileID = os.path.basename(sentinelFile).split('_')[0]
    alignedOverlap = extract_overlap(label, tile, tileID)
    blue = tile.read(1).astype('float32')
    green = tile.read(2).astype('float32')
    red = tile.read(3).astype('float32')
    nir = tile.read(7).astype('float32')
    swir = tile.read(8).astype('float32')
    ndvi = (nir - red) / (nir + red)
    ndwi = (green - nir) / (green + nir)
    evi = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))
    ndbi = (swir - nir) / (swir + nir)
    mndwi = (green - swir) / (green + swir)
    indicesDict = {
        'NDVI': ndvi,
        'NDWI': ndwi,
        'EVI': evi,
        'NDBI': ndbi,
        'MNDWI': mndwi
    }
    blockSize = 1024
    numRows, numCols = alignedOverlap.shape
    columns = ['Label']

    for name in indicesDict.keys():
        columns.extend([name, f'{name} mean', f'{name} variance'])

    columns.append('Water')

    for rowStart in range(0, numRows, blockSize):
        for colStart in range(0, numCols, blockSize):
            rowEnd = min(rowStart + blockSize, numRows)
            colEnd = min(colStart + blockSize, numCols)
            alignedBlock = alignedOverlap[rowStart:rowEnd, colStart:colEnd]
            indicesBlock = {}

            for name, arr in indicesDict.items():
                indicesBlock[name] = arr[rowStart:rowEnd, colStart:colEnd]

            featuresBlock = extractRawVals(alignedBlock, indicesBlock, 3)
            createCSV(featuresBlock, columns, True if rowStart == 0 and colStart == 0 else False)