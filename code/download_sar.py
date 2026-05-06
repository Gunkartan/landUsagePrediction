import ee

ee.Authenticate()
ee.Initialize()
gaul = ee.FeatureCollection('FAO/GAUL/2015/level1')
rayong = gaul.filter(ee.Filter.eq('ADM1_NAME', 'Rayong')).geometry()
s = (ee.ImageCollection('COPERNICUS/S1_GRD')
     .filterBounds(rayong)
     .filterDate('2018-10-01', '2018-10-31')
     .filter(ee.Filter.eq('instrumentMode', 'IW'))
     .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
     .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
     .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
     .select(['VV', 'VH'])
     .sort('system:time_start'))

if s.size().getInfo() == 0:
    raise ValueError('No Sentinel-1 images found for this date or region')

image = s.median().clip(rayong)
filtered_linear = image.focal_mean(radius=10, units='meters')
filtered_linear = filtered_linear.max(1e-10)
filtered_db = filtered_linear.log10().multiply(10)
ee.batch.Export.image.toDrive(
    image=filtered_db,
    description='rayong_sar',
    folder='GEE',
    fileNamePrefix='rayong_sar',
    region=rayong,
    scale=10,
    crs='EPSG:32647',
    maxPixels=1e13
).start()
print('Done')