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
    re_ndvi_oct = (nir_oct - re_early_oct) / (nir_oct + re_early_oct) #Red-edge NDVI.
    ndre_oct = (nir_oct - re_mid_oct) / (nir_oct + re_mid_oct) #Normalized difference red-edge.
    gci_oct = nir_oct / green_oct - 1 #Green chlorophyll index.
    ndmi_oct = (nir_oct - swir_oct) / (nir_oct + swir_oct) #Normalized difference moisture index.
    gndvi_oct = (nir_oct - green_oct) / (nir_oct + green_oct) #Green normalized difference vegetation index.
    savi_oct = ((1 + 0.5) * (nir_oct - red_oct)) / (nir_oct + red_oct + 0.5) #Soil adjusted vegetation index.
    tgi_oct = -0.5 * (190 * (red_oct - green_oct) - 120 * (red_oct - blue_oct)) #Triangular greenness index.
    arvi_oct = (nir_oct - (2 * red_oct - blue_oct)) / (nir_oct + (2 * red_oct - blue_oct)) #Atmospherically resistant vegetation index.
    nbr_oct = (nir_oct - swir_long_oct) / (nir_oct + swir_long_oct) #Normalized burn ratio.
    vari_oct = (green_oct - red_oct) / (green_oct + red_oct - blue_oct) #Visible atmospherically resistant index.
    ci_oct = nir_oct / re_early_oct - 1 #Chlorophyll index.
    bsi_oct = ((swir_oct + red_oct) - (nir_oct + blue_oct)) / ((swir_oct + red_oct) + (nir_oct + blue_oct)) #Bare soil index.
    evi_no_blue_oct = 2.5 * (nir_oct - red_oct) / (nir_oct + 2.4 * red_oct + 1)
    sr_oct = nir_oct / red_oct #Simple ratio.
    dvi_oct = nir_oct - red_oct #Difference vegetation index.
    ndvi_narrow_oct = (nir_narrow_oct - red_oct) / (nir_narrow_oct + red_oct)

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
    re_ndvi_nov = (nir_nov - re_early_nov) / (nir_nov + re_early_nov)
    ndre_nov = (nir_nov - re_mid_nov) / (nir_nov + re_mid_nov)
    gci_nov = nir_nov / green_nov - 1
    ndmi_nov = (nir_nov - swir_nov) / (nir_nov + swir_nov)
    gndvi_nov = (nir_nov - green_nov) / (nir_nov + green_nov)
    savi_nov = ((1 + 0.5) * (nir_nov - red_nov)) / (nir_nov + red_nov + 0.5)
    tgi_nov = -0.5 * (190 * (red_nov - green_nov) - 120 * (red_nov - blue_nov))
    arvi_nov = (nir_nov - (2 * red_nov - blue_nov)) / (nir_nov + (2 * red_nov - blue_nov))
    nbr_nov = (nir_nov - swir_long_nov) / (nir_nov + swir_long_nov)
    vari_nov = (green_nov - red_nov) / (green_nov + red_nov - blue_nov)
    ci_nov = nir_nov / re_early_nov - 1
    bsi_nov = ((swir_nov + red_nov) - (nir_nov + blue_nov)) / ((swir_nov + red_nov) + (nir_nov + blue_nov))
    evi_no_blue_nov = 2.5 * (nir_nov - red_nov) / (nir_nov + 2.4 * red_nov + 1)
    sr_nov = nir_nov / red_nov
    dvi_nov = nir_nov - red_nov
    ndvi_narrow_nov = (nir_narrow_nov - red_nov) / (nir_narrow_nov + red_nov)

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
    re_ndvi_dec = (nir_dec - re_early_dec) / (nir_dec + re_early_dec)
    ndre_dec = (nir_dec - re_mid_dec) / (nir_dec + re_mid_dec)
    gci_dec = nir_dec / green_dec - 1
    ndmi_dec = (nir_dec - swir_dec) / (nir_dec + swir_dec)
    gndvi_dec = (nir_dec - green_dec) / (nir_dec + green_dec)
    savi_dec = ((1 + 0.5) * (nir_dec - red_dec)) / (nir_dec + red_dec + 0.5)
    tgi_dec = -0.5 * (190 * (red_dec - green_dec) - 120 * (red_dec - blue_dec))
    arvi_dec = (nir_dec - (2 * red_dec - blue_dec)) / (nir_dec + (2 * red_dec - blue_dec))
    nbr_dec = (nir_dec - swir_long_dec) / (nir_dec + swir_long_dec)
    vari_dec = (green_dec - red_dec) / (green_dec + red_dec - blue_dec)
    ci_dec = nir_dec / re_early_dec - 1
    bsi_dec = ((swir_dec + red_dec) - (nir_dec + blue_dec)) / ((swir_dec + red_dec) + (nir_dec + blue_dec))
    evi_no_blue_dec = 2.5 * (nir_dec - red_dec) / (nir_dec + 2.4 * red_dec + 1)
    sr_dec = nir_dec / red_dec
    dvi_dec = nir_dec - red_dec
    ndvi_narrow_dec = (nir_narrow_dec - red_dec) / (nir_narrow_dec + red_dec)

    indices_dict = {
        'ndvi_oct': ndvi_oct,
        'ndwi_oct': ndwi_oct,
        'evi_oct': evi_oct,
        'ndbi_oct': ndbi_oct,
        're_ndvi_oct': re_ndvi_oct,
        'ndre_oct': ndre_oct,
        'gci_oct': gci_oct,
        'ndmi_oct': ndmi_oct,
        'gndvi_oct': gndvi_oct,
        'savi_oct': savi_oct,
        'tgi_oct': tgi_oct,
        'arvi_oct': arvi_oct,
        'nbr_oct': nbr_oct,
        'vari_oct': vari_oct,
        'ci_oct': ci_oct,
        'bsi_oct': bsi_oct,
        'evi_no_blue_oct': evi_no_blue_oct,
        'sr_oct': sr_oct,
        'dvi_oct': dvi_oct,
        'ndvi_narrow_oct': ndvi_narrow_oct,
        'ndvi_nov': ndvi_nov,
        'ndwi_nov': ndwi_nov,
        'evi_nov': evi_nov,
        'ndbi_nov': ndbi_nov,
        're_ndvi_nov': re_ndvi_nov,
        'ndre_nov': ndre_nov,
        'gci_nov': gci_nov,
        'ndmi_nov': ndmi_nov,
        'gndvi_nov': gndvi_nov,
        'savi_nov': savi_nov,
        'tgi_nov': tgi_nov,
        'arvi_nov': arvi_nov,
        'nbr_nov': nbr_nov,
        'vari_nov': vari_nov,
        'ci_nov': ci_nov,
        'bsi_nov': bsi_nov,
        'evi_no_blue_nov': evi_no_blue_nov,
        'sr_nov': sr_nov,
        'dvi_nov': dvi_nov,
        'ndvi_narrow_nov': ndvi_narrow_nov,
        'ndvi_dec': ndvi_dec,
        'ndwi_dec': ndwi_dec,
        'evi_dec': evi_dec,
        'ndbi_dec': ndbi_dec,
        're_ndvi_dec': re_ndvi_dec,
        'ndre_dec': ndre_dec,
        'gci_dec': gci_dec,
        'ndmi_dec': ndmi_dec,
        'gndvi_dec': gndvi_dec,
        'savi_dec': savi_dec,
        'tgi_dec': tgi_dec,
        'arvi_dec': arvi_dec,
        'nbr_dec': nbr_dec,
        'vari_dec': vari_dec,
        'ci_dec': ci_dec,
        'bsi_dec': bsi_dec,
        'evi_no_blue_dec': evi_no_blue_dec,
        'sr_dec': sr_dec,
        'dvi_dec': dvi_dec,
        'ndvi_narrow_dec': ndvi_narrow_dec
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