import os
import json
import io
import glob
import argparse

from pandas import read_parquet
from PIL import Image, PngImagePlugin
import lmdb


def convert_parquet(filename, save_dir, jsonl_name, lmdb_env):
    data = read_parquet(filename)
    messages = data["messages"]
    images = data["images"]
    assert len(messages) == len(images)

    # create tar file
    imgs_data = {}
    jsonl_datas = []
    for i in range(len(messages)):
        imgs = images[i]
        imgs_names = []
        for j in range(len(imgs)):
            img_name = f"{i}_{j}.png"
            bytes_io = imgs[j]['bytes']
            pil_img = Image.open(io.BytesIO(bytes_io))
            width, height = pil_img.size
            imgs_names.append(dict(
                image_path=img_name,
                width=width,
                height=height,
            ))
            imgs_data[img_name] = bytes_io
        jsonl_datas.append(dict(conversations=list(messages[i]), images=imgs_names))

    # save jsonl file
    with open(os.path.join(save_dir, jsonl_name), "w") as f:
        for jsonl_data in jsonl_datas:
            f.write(json.dumps(jsonl_data, ensure_ascii=False) + "\n")

    # write to lmdb
    with lmdb_env.begin(write=True) as txn:
        for img_name, bytes_io in imgs_data.items():
            value = txn.put(img_name.encode(), bytes_io)
            assert value


def get_args():
    parser = argparse.ArgumentParser(description="convert the llava dataset")
    parser.add_argument(
        "--input_path", type=str, required=True, help="BUAADreamer/llava-en-zh-300k source path"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="BUAADreamer/llava-en-zh-300k gcore output path"
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    all_parquet_file = glob.glob(os.path.join(args.input_path, '*.parquet'))

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    lmdb_path = os.path.join(args.output_path, "img_file.lmdb")
    lmdb_env = lmdb.open(lmdb_path, map_size=1 * 2**40)  # 1TB
    PngImagePlugin.MAX_TEXT_CHUNK = 1024 * 1024 * 1024  # 1GB

    for filename in all_parquet_file:
        basename = os.path.basename(filename)
        jsonl_filename = basename.replace(".parquet", ".jsonl")
        convert_parquet(filename, args.output_path, jsonl_filename, lmdb_env)
    lmdb_env.close()
