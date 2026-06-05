import os
import time
import json
import argparse
from tqdm import tqdm
from typing import Tuple, List, Dict

from fastapi import FastAPI
import uvicorn
import asyncio
import lmdb
import requests
import threading
from PIL import Image
from mpi4py import MPI


def convert_conversation_v1(src_data: dict) -> dict:
    dst_data = dict()
    dst_data["conversations"] = []
    if "system" in src_data:
        dst_data["conversations"].append(dict(role="system", content=src_data["system"]))
    dst_data["conversations"].append(dict(role="user", content=src_data["query"]))
    dst_data["conversations"].append(dict(role="assistant", content=src_data["response"]))
    return dst_data


def convert_image_v1(src_data: dict, root_dir: str) -> Tuple[bool, List[dict]]:
    scr_images = src_data["images"]
    dst_images = []
    for image in scr_images:
        try:
            # test the image is valid
            image_obj = Image.open(os.path.join(root_dir, image))
            image_obj = image_obj.convert("RGB")
        except:
            print(f"image {image} is invalid, abort this record")
            return (False, {})
        dst_images.append(dict(image_path=image))

    return (True, dst_images)


def convert_jsonl(
    jsonl_filepath: str, save_dir: str, rank: int, world_size: int, images_dir: str = None
):
    if images_dir is not None:
        root_dir = images_dir
    else:
        root_dir = os.path.dirname(jsonl_filepath) if os.path.isabs(jsonl_filepath) else os.getcwd()
    new_jsonl_name = f".rank-{rank}-" + os.path.basename(jsonl_filepath)
    jsonl_f = open(os.path.join(save_dir, new_jsonl_name), 'w')

    src_lines = open(jsonl_filepath, 'r').readlines()
    rank_size = (len(src_lines) + world_size - 1) // world_size
    begin_index = rank * rank_size
    end_index = min((rank + 1) * rank_size, len(src_lines))
    src_lines = src_lines[begin_index:end_index]
    dst_lines = []
    for src_line in tqdm(src_lines, desc='convert json'):
        src_data = json.loads(src_line)
        dst_data = convert_conversation_v1(src_data)
        res, images = convert_image_v1(src_data, root_dir)

        # 样本不合法，跳过
        if not res:
            continue
        dst_data["images"] = images
        dst_lines.append(dst_data)

    jsonl_f.write("\n".join([json.dumps(line, ensure_ascii=False) for line in dst_lines]))
    jsonl_f.write("\n")
    jsonl_f.close()


def merge_all_jsonl(jsonl_filepath: str, save_dir: str, world_size: int):
    merge_jsonl_name = os.path.basename(jsonl_filepath)
    jsonl_merge_f = open(os.path.join(save_dir, merge_jsonl_name), 'w')

    for rank in tqdm(range(world_size), desc='merge json'):
        new_jsonl_name = f".rank-{rank}-" + os.path.basename(jsonl_filepath)
        with open(os.path.join(save_dir, new_jsonl_name), 'r') as f:
            jsonl_merge_f.write(f.read())
        os.remove(os.path.join(save_dir, new_jsonl_name))

    jsonl_merge_f.close()


def write_lmdb(
    jsonl_filepath: str, save_dir: str, rank: int, world_size: int, images_dir: str = None
):
    if images_dir is not None:
        root_dir = images_dir
    else:
        root_dir = os.path.dirname(jsonl_filepath) if os.path.isabs(jsonl_filepath) else os.getcwd()

    image_files = set()
    merge_jsonl_name = os.path.basename(jsonl_filepath)
    jsonl_merge_fullpath = os.path.join(save_dir, merge_jsonl_name)
    with open(jsonl_merge_fullpath, 'r') as f:
        for line in f.readlines():
            imgs = json.loads(line)["images"]
            for img in imgs:
                image_files.add(img['image_path'])
    image_files = list(image_files)
    image_files.sort()

    rank_size = (len(image_files) + world_size - 1) // world_size
    begin_index = rank * rank_size
    end_index = min((rank + 1) * rank_size, len(image_files))
    image_files = image_files[begin_index:end_index]

    post_json = {}
    for img_file in tqdm(image_files, desc='write to lmdb'):
        with open(os.path.join(root_dir, img_file), 'rb') as f:
            value = f.read()
        post_json.update({img_file: value.decode('latin1')})
        if len(post_json) >= 32:
            response = requests.post(f"http://localhost:8223/write", json=post_json, timeout=60 * 5)
            assert response.status_code == 200
            assert response.json()['data']
            post_json = {}

    if len(post_json) > 0:
        response = requests.post(f"http://localhost:8223/write", json=post_json, timeout=60 * 5)
        assert response.status_code == 200
        assert response.json()['data']


def run_server(lmdb_env):
    app = FastAPI()

    @app.get("/live")
    async def live():
        return {'data': True}

    @app.post("/write")
    async def write(kv: Dict[str, str]):
        res = True
        with lmdb_env.begin(write=True) as txn:
            for key, value in kv.items():
                value = txn.put(key.encode(), value.encode('latin1'))
                if not value:
                    res = False
                    break

        return {"data": res}

    uvicorn.run(app, host="localhost", port=8223, log_level="error")


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='多模型样本处理工具脚本')
    group.add_argument('--jsonl_filepath', type=str, required=True, help="jsonl文件路径")
    group.add_argument('--images_dir', type=str, default=None, help="jsonl内图片路径的目录")
    group.add_argument('--save_dir', type=str, required=True, help="保存的文件路径")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    comm = MPI.COMM_WORLD  # 假定使用 mpi，其实用 torchrun 也可以，但是 mpi 更加方便。
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(f"begin preprocess {rank} {size}", flush=True)
    print(f"inspect args  {args}", flush=True)
    comm.Barrier()

    convert_jsonl(args.jsonl_filepath, args.save_dir, rank, size, args.images_dir)
    comm.Barrier()

    if rank == 0:
        print("merge all jsonl file")
        merge_all_jsonl(args.jsonl_filepath, args.save_dir, size)
    comm.Barrier()

    # 启动一个 webserver
    if rank == 0:
        # 40T
        lmdb_env = lmdb.open(os.path.join(args.save_dir, "img_file.lmdb"), map_size=10 * 2**40)

        thread = threading.Thread(target=run_server, args=(lmdb_env, ))
        thread.daemon = True
        thread.start()
        # 等待 webserver 启动成功
        for _ in range(10):
            try:
                response = requests.get("http://localhost:8223/live")
                if response.status_code == 200:
                    break
            except Exception as e:
                pass
            time.sleep(1)
    comm.Barrier()
    write_lmdb(args.jsonl_filepath, args.save_dir, rank, size, args.images_dir)
    comm.Barrier()
    if rank == 0:
        lmdb_env.close()
    comm.Barrier()
