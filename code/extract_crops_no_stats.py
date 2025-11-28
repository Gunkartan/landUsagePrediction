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
    msi_oct = swir_oct / nir_oct
    gvmi_oct = ((nir_oct + 0.1) - (swir_long_oct + 0.02)) / ((nir_oct + 0.1) + (swir_long_oct + 0.02))
    gndvi_oct = (nir_oct - green_oct) / (nir_oct + green_oct)
    rendvi_oct = (nir_oct - re_early_oct) / (nir_oct + re_early_oct)
    evi_no_blue_oct = 2.5 * (nir_oct - red_oct) / (nir_oct + 2.4 * red_oct + 1)
    rvi_oct = nir_oct / red_oct
    sbi_oct = np.sqrt((blue_oct ** 2 + green_oct ** 2 + red_oct ** 2 + nir_oct ** 2) / 4)
    b_oct = 0.3037 * blue_oct + 0.2793 * green_oct + 0.4743 * red_oct + 0.5585 * nir_oct + 0.5082 * swir_oct + 0.1863 * swir_long_oct
    vbi_oct = (blue_oct + green_oct + red_oct) / 3
    swirbi_oct = (swir_oct + swir_long_oct) / 2
    rgr_oct = red_oct / green_oct
    rnr_oct = red_oct / nir_oct
    grer_oct = green_oct / re_early_oct
    snr_oct = swir_oct / nir_oct

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
    msi_nov = swir_nov / nir_nov
    gvmi_nov = ((nir_nov + 0.1) - (swir_long_nov + 0.02)) / ((nir_nov + 0.1) + (swir_long_nov + 0.02))
    gndvi_nov = (nir_nov - green_nov) / (nir_nov + green_nov)
    rendvi_nov = (nir_nov - re_early_nov) / (nir_nov + re_early_nov)
    evi_no_blue_nov = 2.5 * (nir_nov - red_nov) / (nir_nov + 2.4 * red_nov + 1)
    rvi_nov = nir_nov / red_nov
    sbi_nov = np.sqrt((blue_nov ** 2 + green_nov ** 2 + red_nov ** 2 + nir_nov ** 2) / 4)
    b_nov = 0.3037 * blue_nov + 0.2793 * green_nov + 0.4743 * red_nov + 0.5585 * nir_nov + 0.5082 * swir_nov + 0.1863 * swir_long_nov
    vbi_nov = (blue_nov + green_nov + red_nov) / 3
    swirbi_nov = (swir_nov + swir_long_nov) / 2
    rgr_nov = red_nov / green_nov
    rnr_nov = red_nov / nir_nov
    grer_nov = green_nov / re_early_nov
    snr_nov = swir_nov / nir_nov

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
    msi_dec = swir_dec / nir_dec
    gvmi_dec = ((nir_dec + 0.1) - (swir_long_dec + 0.02)) / ((nir_dec + 0.1) + (swir_long_dec + 0.02))
    gndvi_dec = (nir_dec - green_dec) / (nir_dec + green_dec)
    rendvi_dec = (nir_dec - re_early_dec) / (nir_dec + re_early_dec)
    evi_no_blue_dec = 2.5 * (nir_dec - red_dec) / (nir_dec + 2.4 * red_dec + 1)
    rvi_dec = nir_dec / red_dec
    sbi_dec = np.sqrt((blue_dec ** 2 + green_dec ** 2 + red_dec ** 2 + nir_dec ** 2) / 4)
    b_dec = 0.3037 * blue_dec + 0.2793 * green_dec + 0.4743 * red_dec + 0.5585 * nir_dec + 0.5082 * swir_dec + 0.1863 * swir_long_dec
    vbi_dec = (blue_dec + green_dec + red_dec) / 3
    swirbi_dec = (swir_dec + swir_long_dec) / 2
    rgr_dec = red_dec / green_dec
    rnr_dec = red_dec / nir_dec
    grer_dec = green_dec / re_early_dec
    snr_dec = swir_dec / nir_dec

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
        'msi_oct': msi_oct,
        'gvmi_oct': gvmi_oct,
        'gndvi_oct': gndvi_oct,
        'rendvi_oct': rendvi_oct,
        'evi_no_blue_oct': evi_no_blue_oct,
        'rvi_oct': rvi_oct,
        'sbi_oct': sbi_oct,
        'b_oct': b_oct,
        'vbi_oct': vbi_oct,
        'swirbi_oct': swirbi_oct,
        'rgr_oct': rgr_oct,
        'rnr_oct': rnr_oct,
        'grer_oct': grer_oct,
        'snr_oct': snr_oct,
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
        'msi_nov': msi_nov,
        'gvmi_nov': gvmi_nov,
        'gndvi_nov': gndvi_nov,
        'rendvi_nov': rendvi_nov,
        'evi_no_blue_nov': evi_no_blue_nov,
        'rvi_nov': rvi_nov,
        'sbi_nov': sbi_nov,
        'b_nov': b_nov,
        'vbi_nov': vbi_nov,
        'swirbi_nov': swirbi_nov,
        'rgr_nov': rgr_nov,
        'rnr_nov': rnr_nov,
        'grer_nov': grer_nov,
        'snr_nov': snr_nov,
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
        'ndii_dec': ndii_dec,
        'msi_dec': msi_dec,
        'gvmi_dec': gvmi_dec,
        'gndvi_dec': gndvi_dec,
        'rendvi_dec': rendvi_dec,
        'evi_no_blue_dec': evi_no_blue_dec,
        'rvi_dec': rvi_dec,
        'sbi_dec': sbi_dec,
        'b_dec': b_dec,
        'vbi_dec': vbi_dec,
        'swirbi_dec': swirbi_dec,
        'rgr_dec': rgr_dec,
        'rnr_dec': rnr_dec,
        'grer_dec': grer_dec,
        'snr_dec': snr_dec
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