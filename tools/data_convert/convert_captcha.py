import os
import glob
import json
import io
import argparse

from pandas import read_parquet
from PIL import Image
import lmdb


def convert_conversation():
    conversation = []
    # add system prompt
    conversation.append(dict(role="system", content="You are a helpful assistant."))
    conversation.append(
        dict(role="user", content=f"<image>Put the captcha of the image within \\boxed{{}}")
    )
    return conversation


def convert_parquet(filename, save_dir, jsonl_name, basename):
    os.makedirs(save_dir, exist_ok=True)
    jsonl_f = open(os.path.join(save_dir, jsonl_name), "w")
    lmdb_path = os.path.join(save_dir, "img_file.lmdb")
    lmdb_env = lmdb.open(lmdb_path, map_size=1 * 2**40)  # 1TB
    txn = lmdb_env.begin(write=True)

    data = read_parquet(filename)

    labels = data["label"]
    images = data["image"]
    assert len(labels) == len(images)

    for i in range(len(images)):
        label = labels[i]
        img_bytes = images[i]['bytes']
        conversation = convert_conversation()

        img_name = os.path.join(basename, f"{i}.png")
        value = txn.put(img_name.encode(), img_bytes)
        pil_img = Image.open(io.BytesIO(img_bytes))
        width, height = pil_img.size
        assert value
        imgs_names = [dict(
            image_path=img_name,
            width=width,
            height=height,
        )]
        one_line = dict(
            conversations=conversation,
            label=label,
            images=imgs_names,
        )
        jsonl_f.write(json.dumps(one_line, ensure_ascii=True) + "\n")
    txn.commit()
    lmdb_env.close()


def get_args():
    parser = argparse.ArgumentParser(description="convert the llava dataset")
    parser.add_argument(
        "--input_path", type=str, required=True, help="yusuf802/captcha_dataset source path"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="yusuf802/captcha_dataset gcore output path"
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    all_files = glob.glob(os.path.join(args.input_path, '*.parquet'))
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    for filename in all_files:
        basename = os.path.basename(filename)
        jsonl_filename = basename.replace(".parquet", ".jsonl")
        convert_parquet(filename, args.output_path, jsonl_filename, basename)
