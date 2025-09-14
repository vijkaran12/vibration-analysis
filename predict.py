import requests

url = "https://vibration-analysis-rwy8.onrender.com"
data = {"data": [0.23, 0.87, 0.12, 3.45]}
resp = requests.post(url, json=data)
print(resp.status_code, resp.json())
