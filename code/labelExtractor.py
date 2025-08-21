import os.path
import numpy as np
import rasterio
from rasterio.windows import from_bounds

def extract_overlap(label_raster, tile_raster, tile_id, targetLabels):
    label_bounds = label_raster.bounds
    tile_bounds = tile_raster.bounds

    intersection = (max(label_bounds.left, tile_bounds.left),
                    max(label_bounds.bottom, tile_bounds.bottom),
                    min(label_bounds.right, tile_bounds.right),
                    min(label_bounds.top, tile_bounds.top))

    if intersection[0] < intersection[2] and intersection[1] < intersection[3]:

        label_window = from_bounds(*intersection, transform=label_raster.transform)
        overlap_data = label_raster.read(1, window=label_window)
        maskMisc = ~np.isin(overlap_data, targetLabels)
        overlap_data[maskMisc] = 1391

        print(overlap_data.shape)
        # Calculate the window in the small raster
        raster_window = from_bounds(*intersection, transform=tile_raster.transform)

        # Prepare an output array matching the small raster's size
        aligned_overlap = np.full((tile_raster.height, tile_raster.width), 0,
                                  dtype=label_raster.dtypes[0])

        # Calculate row/column offsets in the small raster
        row_start = int(raster_window.row_off)
        row_end = row_start + overlap_data.shape[0]
        col_start = int(raster_window.col_off)
        col_end = col_start + overlap_data.shape[1]

        # Embed the overlap data into the corresponding location
        aligned_overlap[row_start:row_end, col_start:col_end] = overlap_data

        # Write the aligned overlap to a new raster
        profile = tile_raster.profile
        profile.update({
            'driver': 'GTiff',
            'count': 1,
            'nodata': 0,
            'compress': 'LZW',
        })
        os.makedirs("./label/", exist_ok=True)
        with rasterio.open(f"./label/label_{tile_id}.tif", 'w', **profile) as dest:
            dest.write(aligned_overlap, 1)

    return aligned_overlap


if __name__ == "__main__":
    label_file = "cmi_raster_67.tif"
    sentinel_files = [
        "./S2_data/S2B_MSIL2A_20210616T035539_N0500_R004_T47QMA_20230322T123949.tif",
        "./S2_data/S2B_MSIL2A_20210623T034539_N0500_R104_T47QMV_20230129T154153.tif",
        "./S2_data/S2B_MSIL2A_20210623T034539_N0500_R104_T47QNB_20230129T154153.tif",
        "./S2_data/S2B_MSIL2A_20210626T035539_N0500_R004_T47QMB_20230202T075333.tif",
        "./S2_data/S2B_MSIL2A_20210626T035539_N0500_R004_T47QNC_20230202T075333.tif",
        "./S2_data/S2B_MSIL2A_20230623T034539_N0509_R104_T47QNA_20230623T080518.tif",
    ]

    label = rasterio.open(label_file)
    for file in sentinel_files:
        tile = rasterio.open(file)
        tile_id = os.path.basename(file).split("_")[5][1:]
        extract_overlap(label, tile, tile_id)
        tile.close()