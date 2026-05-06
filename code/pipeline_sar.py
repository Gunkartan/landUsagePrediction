import os
import subprocess
import numpy as np
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from download_sar_snap import *
from extract_sar import *

def process_with_snap(input_file):
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(processed_dir, base_name + '.tif')

    if os.path.exists(output_file):
        print(f'Skipping already processed {input_file}')

        return output_file
    
    gpt_path = r'C:\Program Files\esa-snap\bin\gpt.exe'
    cmd = [
        gpt_path,
        graph,
        f'-Pinput={input_file}',
        f'-Poutput={output_file}'
    ]
    subprocess.run(cmd, check=True)

    return output_file

def process_all():
    processed_files = []

    for file in sorted(os.listdir(input_dir)):
        if file.endswith('.SAFE'):
            path = os.path.join(input_dir, file)
            print(f'Processing {file}')
            out = process_with_snap(path)
            processed_files.append(out)

    return processed_files

def compute_median(files):
    stack = []
    profile = None

    for f in files:
        with rasterio.open(f) as src:
            if profile is None:
                profile = src.profile.copy()

            stack.append(src.read())

    stack = np.stack(stack, axis=0)
    median = np.median(stack, axis=0)

    return median, profile

def save_median(median, profile):
    profile.update(
        dtype=rasterio.float32,
        count=median.shape[0]
    )

    with rasterio.open('median.tif', 'w', **profile) as dst:
        dst.write(median.astype(np.float32))

def clip_to_region():
    gdf = gpd.read_file(shapefile)

    with rasterio.open('median.tif') as src:
        gdf = gdf.to_crs(src.crs)
        out_image, out_transform = mask(src, gdf.geometry, crop=True)
        profile = src.profile
        profile.update({
            'height': out_image.shape[1],
            'width': out_image.shape[2],
            'transform': out_transform,
            'count': out_image.shape[0],
            'dtype': out_image.dtype
        })

        with rasterio.open('median_clipped.tif', 'w', **profile) as dst:
            dst.write(out_image)

if __name__ == '__main__':
    input_dir = 'raw'
    processed_dir = 'processed'
    graph = '../graph/graph_sar.xml'
    shapefile = '../shapefiles/LU_RYG_2561.shp'
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    download_sar_snap()
    extract_sar()
    files = process_all()

    if not files:
        raise ValueError('No processed files found')
    
    median, profile = compute_median(files)
    save_median(median, profile)
    clip_to_region()
    print('Pipeline complete')