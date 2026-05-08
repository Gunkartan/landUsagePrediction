import os
import time
import requests
from tqdm import tqdm
from credential import *

class CopernicusSession:
    def __init__(self):
        self.token = None
        self.expires_at = 0

    def refresh_token(self):
        token_data = get_access_token()
        self.token = token_data['access_token']
        expires_in = token_data.get('expires_in', 1800)
        self.expires_at = time.time() + max(expires_in - 60, 60)

    def headers(self):
        if self.token is None or time.time() >= self.expires_at:
            self.refresh_token()

        return {
            'Authorization': f'Bearer {self.token}',
            'User-Agent': 'Mozilla/5.0'
        }
    
    def get(self, url, *, params=None, stream=False, timeout=60):
        last_response = None

        for attempt in range(3):
            response = requests.get(
                url,
                headers=self.headers(),
                params=params,
                stream=stream,
                timeout=timeout
            )
            last_response = response

            if response.status_code == 401:
                response.close()
                self.refresh_token()

                continue

            if response.status_code not in (403, 429, 500, 502, 503, 504):
                return response
            
            if attempt < 2:
                response.close()
                time.sleep(5 * (attempt + 1))

        return last_response

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

    return response.json()

def raise_for_status_with_context(response):
    try:
        response.raise_for_status()

    except requests.HTTPError as exc:
        if response.status_code == 403:
            raise requests.HTTPError(
                'Copernicus returned 403, so please wait a few minutes and then retry.'
            ) from exc
        
        if response.status_code == 401:
            raise requests.HTTPError(
                'Copernicus returned 401, so please verify that the username and password are correct.'
            ) from exc
        
        raise

def download_sar_snap():
    output_dir = raw_dir
    os.makedirs(output_dir, exist_ok=True)
    session = CopernicusSession()
    search_filter = (
        'Collection/Name eq "SENTINEL-1" and '
        'Attributes/OData.CSC.StringAttribute/any('
        'att:att/Name eq "productType" and '
        'att/OData.CSC.StringAttribute/Value eq "IW_GRDH_1S") and '
        'OData.CSC.Intersect('
        f'area=geography"SRID=4326;{rayong_polygon}") and '
        'ContentDate/Start gt 2018-10-01T00:00:00.000Z and '
        'ContentDate/Start lt 2018-10-31T23:59:59.000Z'
    )
    params = {
        '$filter': search_filter,
        '$orderby': 'ContentDate/Start asc',
        '$top': 20
    }
    response = session.get(
        catalogue_url,
        params=params
    )
    raise_for_status_with_context(response)
    data = response.json()
    products = [
        product for product in data.get('value', [])
        if '_IW_GRDH_1S' in product.get('Name', '')
    ]
    print(f'Found {len(products)} products')

    for product in products:
        product_id = product['Id']
        name = product['Name']
        safe_name = name if name.endswith('.SAFE') else f'{name}.SAFE'
        zip_name = safe_name[:-5]
        download_url = (
            f'{download_url}({product_id})/$value'
        )
        out_path = os.path.join(
            output_dir,
            f'{zip_name}.zip'
        )
        part_path = f'{out_path}.part'
        safe_path = os.path.join(output_dir, safe_name)

        if os.path.exists(safe_path) or os.path.exists(out_path):
            print(f'Skipping {safe_name}')

            continue

        print(f'Downloading {safe_name}')

        with session.get(
            download_url,
            stream=True,
            timeout=60
        ) as r:
            raise_for_status_with_context(r)
            total_size = int(
                r.headers.get('content-length', 0)
            )

            with open(part_path, 'wb') as f, tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc=safe_name
            ) as p_bar:
                for chunk in r.iter_content(
                    chunk_size=8192
                ):
                    if chunk:
                        f.write(chunk)
                        p_bar.update(len(chunk))

        os.replace(part_path, out_path)

if __name__ == '__main__':
    catalogue_url = (
        'https://catalogue.dataspace.copernicus.eu/'
        'odata/v1/Products'
    )
    download_url = (
        'https://download.dataspace.copernicus.eu/'
        'odata/v1/Products'
    )
    rayong_polygon = (
        'POLYGON((101.0 12.5, 101.5 12.5, 101.5 13.0, '
        '101.0 13.0, 101.0 12.5))'
    )
    raw_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'raw')