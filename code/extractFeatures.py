import os.path
import rasterio
import numpy as np
from labelExtractor import extract_overlap
from collections import defaultdict

def computeStatVals(targetLabels, alignedOverlaps, ndvis, ndwis, evis, ndbis, numStatVals, numIndexes, numMonths):
    statDict = defaultdict(list)

    for targetLabel in targetLabels:
        filteredIndexes = [[np.where(alignedOverlap == targetLabel, ndvi, np.nan) for alignedOverlap, ndvi in zip(alignedOverlaps, ndvis)],
                           [np.where(alignedOverlap == targetLabel, ndwi, np.nan) for alignedOverlap, ndwi in zip(alignedOverlaps, ndwis)],
                           [np.where(alignedOverlap == targetLabel, evi, np.nan) for alignedOverlap, evi in zip(alignedOverlaps, evis)],
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

if __name__ == '__main__':
    years = [2018, 2020, 2024]
    dates = ['10-31', '11-30', '12-31']
    numStatVals = 4
    numIndexes = 4
    numMonths = 3

    for year in years:
        labelFile = f'../rasterized/{year}.tif'
        sentinelFiles = [f'../raw/47PQQ_{year}-{date}.tif' for date in dates]
        label = rasterio.open(labelFile)
        targetLabels = {2101: 'ข้าว', 2204: 'มันสำปะหลัง', 2205: 'สัปปะรด',
                        2302: 'ยางพารา', 2303: 'ปาล์มน้ำมัน', 2403: 'ทุเรียน',
                        2404: 'เงาะ', 2405: 'มะพร้าว', 2407: 'มะม่วง',
                        2413: 'ลำไย', 2416: 'ขนุน', 2419: 'มังคุด',
                        2420: 'ลางสาด ลองกอง', 4201: 'น้ำ'}
        tiles = [rasterio.open(sentinelFile) for sentinelFile in sentinelFiles]
        tileID = os.path.basename(sentinelFiles[0]).split('_')[0][0:]
        alignedOverlaps = [extract_overlap(label, tile, tileID) for tile in tiles]
        blues = [tile.read(1).astype('float32') for tile in tiles]
        greens = [tile.read(2).astype('float32') for tile in tiles]
        reds = [tile.read(3).astype('float32') for tile in tiles]
        nirs = [tile.read(7).astype('float32') for tile in tiles]
        swirs = [tile.read(8).astype('float32') for tile in tiles]
        ndvis = [(red - nir) / (red + nir) for red, nir in zip(reds, nirs)]
        ndwis = [(green - nir) / (green + nir) for green, nir in zip(greens, nirs)]
        evis = [2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1) for nir, red, blue in zip(nirs, reds, blues)]
        ndbis = [(swir - nir) / (swir + nir) for swir, nir in zip(swirs, nirs)]
        statDict = computeStatVals(targetLabels, alignedOverlaps, ndvis, ndwis, evis, ndbis, numStatVals, numIndexes, numMonths)

        for targetLabel in statDict:
            print([round(float(num), 3) for num in statDict[targetLabel]])