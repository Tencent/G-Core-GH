import os
import glob
import json
import io
import argparse

from pandas import read_parquet
import lmdb
from PIL import Image

instruction_following = (
    r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    r"The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."
)


def convert_conversation(problem):
    conversation = []
    # add system prompt
    conversation.append(dict(role="system", content="You are a helpful assistant."))
    conversation.append(dict(role="user", content=problem + " " + instruction_following))
    return conversation


instruction_following_reason = (
    r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    r"The reasoning process MUST BE enclosed within <reason> </reason> tags. The final answer MUST BE put in \boxed{}."
)


def convert_conversation_v2(problem):
    conversation = []
    # add system prompt
    conversation.append(dict(role="system", content="You are a helpful assistant."))
    conversation.append(dict(role="user", content=problem + " " + instruction_following_reason))
    return conversation


def convert_parquet(filename, save_dir, jsonl_name, basename, convert_type):
    os.makedirs(save_dir, exist_ok=True)
    jsonl_f = open(os.path.join(save_dir, jsonl_name), "w")
    lmdb_path = os.path.join(save_dir, "img_file.lmdb")
    lmdb_env = lmdb.open(lmdb_path, map_size=1 * 2**40)  # 1TB
    txn = lmdb_env.begin(write=True)

    data = read_parquet(filename)

    images = data["images"]
    problems = data["problem"]
    answers = data["answer"]
    assert len(problems) == len(images)
    assert len(problems) == len(answers)

    for i in range(len(images)):
        problem = problems[i]
        imgs_tmp = images[i].tolist()
        assert len(imgs_tmp) == 1
        img_bytes = imgs_tmp[0]['bytes']
        pil_img = Image.open(io.BytesIO(img_bytes))
        width, height = pil_img.size
        answer = answers[i]
        if convert_type == "v1":
            conversation = convert_conversation(problem)
        elif convert_type == "v2":
            conversation = convert_conversation_v2(problem)
        else:
            assert False, f"no support this convert type:{convert_type}"

        img_name = os.path.join(basename, f"{i}.png")
        value = txn.put(img_name.encode(), img_bytes)
        assert value
        imgs_names = [dict(
            image_path=img_name,
            width=width,
            height=height,
        )]
        label = json.dumps(dict(
            answer=answer,
            problem=problem,
        ))
        one_line = dict(
            conversations=conversation,
            label=label,
            images=imgs_names,
        )
        jsonl_f.write(json.dumps(one_line, ensure_ascii=True) + "\n")
    txn.commit()
    lmdb_env.close()


def get_args():
    parser = argparse.ArgumentParser(description="convert geometry3k dataset")
    parser.add_argument(
        "--input_path", type=str, required=True, help="hiyouga/geometry3k source path"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="hiyouga/geometry3k gcore output path"
    )
    parser.add_argument(
        "--convert_type",
        type=str,
        default="v1",
        choices=["v1", "v2"],
        help="hiyouga/geometry3k convert conversation convert type"
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    all_files = glob.glob(os.path.join(args.input_path, '*.parquet'))
    for filename in all_files:
        basename = os.path.basename(filename)
        jsonl_filename = basename.replace(".parquet", ".jsonl")
        convert_parquet(filename, args.output_path, jsonl_filename, basename, args.convert_type)
