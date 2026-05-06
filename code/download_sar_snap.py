import requests
import os
from tqdm import tqdm
from credential import *
import zipfile

def download_sar_snap():
    output_dir = 'raw'
    os.makedirs(output_dir, exist_ok=True)
    start_date = '2018-10-01'
    end_date = '2018-10-31'
    footprint = 'POLYGON((101.0 12.5, 101.5 12.5, 101.5 13.0, 101.0 13.0, 101.0 12.5))'
    session = requests.Session()
    session.auth = (username, password)
    base_url = 'https://catalogue.dataspace.copernicus.eu/odata/v1/Products'
    query = (
        f"$filter=Collection/Name eq 'SENTINEL-1' "
        f"and ContentDate/Start ge {start_date}T00:00:00.000Z "
        f"and ContentDate/Start le {end_date}T23:59:59.999Z "
        f"and OData.CSC.Intersects(area=geography'{footprint}') "
        f"and Attributes/OData.CSC.StringAttribute/any(a: a/Name eq 'orbitDirection' and a/Value eq 'ASCENDING') "
        f"and Attributes/OData.CSC.StringAttribute/any(a: a/Name eq 'sensorMode' and a/Value eq 'IW') "
        f"and Attributes/OData.CSC.StringAttribute/any(a: a/Name eq 'polarisationChannels' and a/Value eq 'VV VH') "
        f"and Attributes/OData.CSC.StringAttribute/any(a: a/Name eq 'productType' and a/Value eq 'GRD')"
    )
    products = []
    next_url = base_url
    params = {'$filter': query}

    while next_url:
        if params:
            response = session.get(next_url, params=params, timeout=30)

        else:
            response = session.get(next_url, timeout=30)

        response.raise_for_status()
        data = response.json()
        products.extend(data.get('value', []))
        next_url = data.get('@odata.nextLink')
        params = None

    print(f'Fetched {len(products)} products')

    for product in products:
        product_id = product['Id']
        name = product['Name']
        download_url = f'https://zipper.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value'
        out_path = os.path.join(output_dir, f'{product_id}.zip')

        if os.path.exists(out_path):
            print(f'Skipping {name}')
            continue

        print(f'Downloading {name}')

        for attempt in range(3):
            try:
                r = session.get(download_url, stream=True, timeout=60)
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))

                if total_size == 0:
                    total_size = None

                break

            except requests.exceptions.RequestException:
                if attempt == 2:
                    raise

                print(f'Retrying {attempt + 1} out of 3')

        with open(out_path, 'wb') as f, tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            desc=name
        ) as p_bar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    p_bar.update(len(chunk))