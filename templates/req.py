import requests

url = 'http://127.0.0.1:5000/predictions'
data = {'reviewText': 'This is a sample review.'}
response = requests.post(url, json=data)

print(response.json())
