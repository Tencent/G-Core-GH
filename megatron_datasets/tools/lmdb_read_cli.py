import io
import requests
import json

import httpx
from PIL import Image


def fetch_images_from_lmdb(images, lmdb_port) -> list[Image.Image]:
    image_paths = [ele['image_path'] for ele in images]
    response = requests.post(f"http://127.0.0.1:{lmdb_port}/read", json=image_paths)
    assert response.status_code == 200
    json_data_list = response.json()['data']
    res_bytes = {}
    for json_data in json_data_list:
        assert json_data["res"], f"fetch_images_from_lmdb error: {json_data=}"
        res_bytes[json_data["key"]] = json_data["value"].encode('latin1')

    res_list = [Image.open(io.BytesIO(res_bytes[key])) for key in image_paths]
    res_list = [img.convert("RGB") for img in res_list]
    return res_list


if __name__ == "__main__":
    # 写入数据
    data = ["train-00001-of-00059/1_0.png", "train-00001-of-00059/0_0.png"]
    response = requests.post("http://localhost:8200/read", json=data)
    assert response.status_code == 200
    json_data_list = response.json()['data']
    for json_data in json_data_list:
        print(json_data["key"], json_data["res"])
        with open("test.png", 'wb') as f:
            f.write(json_data["value"].encode('latin1'))
        break
