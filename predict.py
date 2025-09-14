import requests

url = "https://vibration-analysis-rwy8.onrender.com/predict"
data = {"data":[0.23, 0.87, 0.12, 3.45]}

response = requests.post(url, json=data)

print("Status code:", response.status_code)
print("Response text:", response.text)  # <-- see the raw response
