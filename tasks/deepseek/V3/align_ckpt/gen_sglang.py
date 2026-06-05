import requests

port = 30000
response = requests.post(
    f"http://127.0.0.1:{port}/generate",
    json={
        "text": "李白，字太白",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 100
        },
    },
)

print(response.json())
