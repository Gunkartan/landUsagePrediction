import requests
from credential import *

if __name__ == '__main__':
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
    print(response.status_code)
    print(response.text)