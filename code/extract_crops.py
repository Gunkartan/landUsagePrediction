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
import rasterio
import numpy as np
import pandas as pd
from datetime import datetime

tile_id = '47PQQ'
year = 2018
months = {
    'Oct': '10',
    'Nov': '11',
    'Dec': '12'
}
label_path = f''