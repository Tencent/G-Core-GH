import os
import json
import glob
import io
import argparse

from numpy import ndarray
from pandas import read_parquet
import lmdb
from PIL import Image


def convert_ele(ele):
    if ele['from'] == 'human':
        return dict(role="user", content=ele['value'])
    elif ele['from'] == 'gpt':
        return dict(role="assistant", content=ele['value'])
    else:
        raise NotImplementedError


def convert_conversation(conversation):
    new_conversation = []
    # add system prompt
    new_conversation.append(dict(role="system", content="You are a helpful assistant."))
    for ele in conversation:
        new_conversation.append(convert_ele(ele))
    return new_conversation


def convert_parquet(filename, save_dir, jsonl_name, basename):
    os.makedirs(save_dir, exist_ok=True)
    jsonl_f = open(os.path.join(save_dir, jsonl_name), "w")
    lmdb_path = os.path.join(save_dir, "img_file.lmdb")
    lmdb_env = lmdb.open(lmdb_path, map_size=1 * 2**40)  # 1TB

    data = read_parquet(filename)

    conversations = data["conversations"]
    chosens = data["chosen"]
    rejecteds = data["rejected"]
    images = data["images"]
    assert len(conversations) == len(chosens)
    assert len(conversations) == len(rejecteds)
    assert len(conversations) == len(images)

    for i in range(len(conversations)):
        assert isinstance(images[i], ndarray)
        assert isinstance(conversations[i], ndarray)

        imgs = images[i].tolist()
        conversation = convert_conversation(conversations[i].tolist())
        rejected = convert_ele(rejecteds[i])
        chosen = convert_ele(chosens[i])
        imgs_names = []
        for j in range(len(imgs)):
            img_name = os.path.join(basename, f"{i}_{j}.png")
            bytes_io = imgs[j]['bytes']
            pil_img = Image.open(io.BytesIO(bytes_io))
            width, height = pil_img.size
            # NOTE: 放到外面可能更快
            with lmdb_env.begin(write=True) as txn:
                value = txn.put(img_name.encode(), bytes_io)
                assert value
            imgs_names.append(dict(
                image_path=img_name,
                width=width,
                height=height,
            ))
        one_line = dict(
            conversations=conversation,
            chosen=chosen,
            rejected=rejected,
            images=imgs_names,
        )
        jsonl_f.write(json.dumps(one_line, ensure_ascii=True) + "\n")
    lmdb_env.close()


def get_args():
    parser = argparse.ArgumentParser(description="Build the GDatasetV4 Meta Json File")
    parser.add_argument(
        "--input_path", type=str, required=True, help="llamafactory/RLHF-V source path"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="llamafactory/RLHF-V gcore output path"
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    all_parquet_file = glob.glob(os.path.join(args.input_path, '*.parquet'))
    for filename in all_parquet_file:
        basename = os.path.basename(filename)
        jsonl_filename = basename.replace(".parquet", ".jsonl")
        convert_parquet(filename, args.output_path, jsonl_filename, basename)
