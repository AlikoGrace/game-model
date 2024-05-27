import requests

url = "http://127.0.0.1:8000/predict"
data = {"data": [1, 7, 2, 2, 0, 2, 1, 0, 3, 3, 0, 3, 1, 8, 5, 3, 2, 3, 0, 6, 0, 4, 9, 1, 0.1, 0.1]}  # Example input data

response = requests.post(url, json=data)

try:
    print(response.json())
except requests.exceptions.JSONDecodeError:
    print("Failed to decode JSON response")
    print(response.text)
