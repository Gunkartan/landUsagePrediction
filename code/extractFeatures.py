import os.path
import rasterio
import numpy as np
import pandas as pd
from labelExtractor import extract_overlap
from collections import defaultdict

def computeStatVals(targetLabels, alignedOverlaps, ndvis, ndwis, ndbis, numStatVals, numIndexes, numMonths):
    statDict = defaultdict(list)

    for targetLabel in targetLabels:
        filteredIndexes = [[np.where(alignedOverlap == targetLabel, ndvi, np.nan) for alignedOverlap, ndvi in zip(alignedOverlaps, ndvis)],
                           [np.where(alignedOverlap == targetLabel, ndwi, np.nan) for alignedOverlap, ndwi in zip(alignedOverlaps, ndwis)],
                           [np.where(alignedOverlap == targetLabel, ndbi, np.nan) for alignedOverlap, ndbi in zip(alignedOverlaps, ndbis)]]
        validIndexes = [[filteredIndex[np.isfinite(filteredIndex)] for filteredIndex in filteredIndexes[i]] for i in range(numIndexes)]

        for i in range(numStatVals):
            for j in range(numIndexes):
                for k in range(numMonths):
                    if i == 0:
                        statDict[targetLabel].append(np.mean(validIndexes[j][k]))

                    elif i == 1:
                        statDict[targetLabel].append(np.std(validIndexes[j][k], ddof=1))

                    elif i == 2:
                        statDict[targetLabel].append(np.min(validIndexes[j][k]))

                    else:
                        statDict[targetLabel].append(np.max(validIndexes[j][k]))

    return statDict

def extractRawVals(targetLabels, alignedOverlaps, ndvis, ndwis, ndbis, numPoints):
    rawDict = defaultdict(list)
    monthRange = range(len(ndvis))

    for targetLabel in targetLabels:
        rows, cols = np.where(alignedOverlaps[0] == targetLabel)
        numPoints = numPoints if len(rows) >= numPoints else len(rows)
        idx = np.random.choice(len(rows), size=numPoints, replace=False)
        coords = list(zip(rows[idx], cols[idx]))

        for r, c in coords:
            vals = []
            vals.extend([ndvis[m][r, c] for m in monthRange])
            vals.extend([ndwis[m][r, c] for m in monthRange])
            vals.extend([ndbis[m][r, c] for m in monthRange])
            rawDict[targetLabel].append(vals)

    return rawDict

if __name__ == '__main__':
    np.random.seed(42)
    years = [2018, 2020, 2024]
    months = ['October', 'November', 'December']
    dates = ['10-31', '11-30', '12-31']
    statVals = ['mean', 'STD', 'minimum', 'maximum']
    indexes = ['NDVI', 'NDWI', 'NDBI']
    rawCols = [f'{month} {index}' for index in indexes for month in months]
    statCols = [f'{month} {statVal} {index}' for statVal in statVals for index in indexes for month in months]
    columns = rawCols + statCols + ['Class']
    numStatVals = 4
    numIndexes = 3
    numMonths = 3
    rows = []

    for year in years:
        labelFile = f'../rasterized/{year}.tif'
        sentinelFiles = [f'../raw/47PQQ_{year}-{date}.tif' for date in dates]
        label = rasterio.open(labelFile)
        targetLabels = {2101: 'ข้าว', 2204: 'มันสำปะหลัง', 2205: 'สัปปะรด',
                        2302: 'ยางพารา', 2303: 'ปาล์มน้ำมัน', 2403: 'ทุเรียน',
                        2404: 'เงาะ', 2405: 'มะพร้าว', 2407: 'มะม่วง',
                        2413: 'ลำไย', 2416: 'ขนุน', 2419: 'มังคุด',
                        2420: 'ลางสาด ลองกอง', 4201: 'น้ำ', 1391: 'อื่น ๆ'}
        tiles = [rasterio.open(sentinelFile) for sentinelFile in sentinelFiles]
        tileID = os.path.basename(sentinelFiles[0]).split('_')[0][0:]
        existingLabels = list(targetLabels.keys())
        alignedOverlaps = [extract_overlap(label, tile, tileID, existingLabels) for tile in tiles]
        blues = [tile.read(1).astype('float32') for tile in tiles]
        greens = [tile.read(2).astype('float32') for tile in tiles]
        reds = [tile.read(3).astype('float32') for tile in tiles]
        nirs = [tile.read(7).astype('float32') for tile in tiles]
        swirs = [tile.read(8).astype('float32') for tile in tiles]
        ndvis = [(nir - red) / (nir + red) for nir, red in zip(nirs, reds)]
        ndwis = [(green - nir) / (green + nir) for green, nir in zip(greens, nirs)]
        ndbis = [(swir - nir) / (swir + nir) for swir, nir in zip(swirs, nirs)]
        statDict = computeStatVals(targetLabels, alignedOverlaps, ndvis, ndwis, ndbis, numStatVals, numIndexes, numMonths)
        rawDict = extractRawVals(targetLabels, alignedOverlaps, ndvis, ndwis, ndbis, 100)

        for targetLabel in rawDict:
            for rawVals in rawDict[targetLabel]:
                row = rawVals + statDict[targetLabel] + [targetLabels[targetLabel]]
                rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    df = df.round(3)
    df.to_csv('allYears.csv', index=False)