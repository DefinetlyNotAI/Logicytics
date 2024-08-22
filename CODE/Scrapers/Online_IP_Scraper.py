import requests
import json

def get_my_ip():
    try:
        response = requests.get('https://api.ipify.org')
        return response.text
    except Exception as e:
        print("Failed to get IP:", e)

def get_location_data(ip):
    api_key = API_KEY
    base_url = f'https://api.ipgeolocation.io/ipgeo?apiKey={api_key}&ip={ip}'
    try:
        response = requests.get(base_url)
        return response.json()
    except Exception as e:
        print("Failed to get location data:", e)

if __name__ == '__main__':
    open('location_data.json', 'w').write(json.dumps(get_location_data(get_my_ip()), indent=4))
