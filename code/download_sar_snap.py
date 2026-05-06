import os
import requests
from tqdm import tqdm
from credential import *

def get_access_token():
    url = (
        'https://identity.dataspace.copernicus.eu/'
        'auth/realms/CDSE/protocol/openid-connect/token'
    )
    data = {
        'client_id': 'cdse-public',
        'username': username,
        'password': password,
        'grant_type': 'password'
    }
    response = requests.post(url, data=data)
    response.raise_for_status()

    return response.json()['access_token']

def download_sar_snap():
    output_dir = 'raw'
    os.makedirs(output_dir, exist_ok=True)
    token = get_access_token()
    
    # OData search and download can both use the same authenticated headers
    headers = {
        'Authorization': f'Bearer {token}',
        'User-Agent': 'Mozilla/5.0'
    }
    
    # NEW: Updated to the active OData API endpoint
    url = 'https://catalogue.dataspace.copernicus.eu/odata/v1/Products'
    
    # NEW: The OData query syntax consolidates all parameters into a single $filter string
    odata_filter = (
        "Collection/Name eq 'SENTINEL-1' and "
        "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq 'GRD') and "
        "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'sensorMode' and att/OData.CSC.StringAttribute/Value eq 'IW') and "
        "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'orbitDirection' and att/OData.CSC.StringAttribute/Value eq 'ASCENDING') and "
        "ContentDate/Start gt 2018-10-01T00:00:00.000Z and "
        "ContentDate/Start lt 2018-10-31T23:59:59.999Z and "
        "OData.CSC.Intersects(area=geography'SRID=4326;POLYGON((101.0 12.5, 101.5 12.5, 101.5 13.0, 101.0 13.0, 101.0 12.5))')"
    )
    
    params = {
        '$filter': odata_filter,
        '$top': 20
    }
    
    response = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=60
    )
    response.raise_for_status()
    data = response.json()
    
    # NEW: OData returns the list inside 'value' instead of 'features'
    products = data.get('value', [])
    print(f'Found {len(products)} products')

    for product in products:
        # NEW: OData uses capitalized 'Id' and 'Name'
        product_id = product['Id']
        name = product['Name']
        
        download_url = (
            'https://zipper.dataspace.copernicus.eu'
            f'/odata/v1/Products({product_id})/$value'
        )
        out_path = os.path.join(output_dir, f'{name}.zip')

        if os.path.exists(out_path):
            print(f'Skipping {name}')
            continue

        print(f'Downloading {name}')

        with requests.get(
            download_url,
            headers=headers,
            stream=True
        ) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))

            with open(out_path, 'wb') as f, tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc=name
            ) as p_bar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        p_bar.update(len(chunk))