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
    headers = {
        'Authorization': f'Bearer {token}',
        'User-Agent': 'Mozilla/5.0'
    }
    url = (
        'https://catalogue.dataspace.copernicus.eu/'
        'resto/api/collections/Sentinel1/search.json'
    )
    params = {
        'startDate': '2018-10-01T00:00:00Z',
        'completionDate': '2018-10-31T23:59:59Z',
        'geometry': '{"type": "Polygon", "coordinates": [[[101.0, 12.5], [101.5, 12.5], [101.5, 13.0], [101.0, 13.0], [101.0, 12.5]]]}',
        'productType': 'GRD',
        'sensorMode': 'IW',
        'orbitDirection': 'ASCENDING',
        'maxRecords': 20
    }
    response = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=60
    )
    response.raise_for_status()
    data = response.json()
    products = data.get('features', [])
    print(f'Found {len(products)} products')

    for product in products:
        product_id = product['id']
        name = product['properties']['title']
        download_url = (
            'https://zipper.dataspace.copernicus.eu'
            f'/odata/v1/Products({product_id})/$value'
        )
        out_path = os.path.join(
            output_dir,
            f'{name}.zip'
        )

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
            total_size = int(
                r.headers.get('content-length', 0)
            )

            with open(out_path, 'wb') as f, tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc=name
            ) as p_bar:
                for chunk in r.iter_content(
                    chunk_size=8192
                ):
                    if chunk:
                        f.write(chunk)
                        p_bar.update(len(chunk))