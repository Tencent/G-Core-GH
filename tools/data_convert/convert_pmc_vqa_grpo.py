import os
import csv
import json
import argparse

from PIL import Image
import lmdb

# acc 过高（有约 88%），caption 内是说明及分析基本上就有答案了，所以它也不分析了，直接给出答案，导致没有 think，fmt 不到 5%
g_user_prompt_caption = """You are a medical image analysis specialist. Please follow these steps to answer the multiple-choice question:

1. Analyze the provided medical image and its caption
2. Read the question and all options thoroughly
3. Apply relevant medical knowledge and imaging interpretation principles
4. Eliminate incorrect options through systematic reasoning
5. Select the single best answer from choices A, B, C, or D

You think about the reasoning process as an internal monologue and then provide the final answer.
The reasoning process MUST BE enclosed within <think> </think> tags, then provide ONLY the letter choice inside <answer></answer> tags.


---
IMAGE: <image>
IMAGE CAPTION: {caption}
---
QUESTION: {question}
OPTIONS:
{A}
{B}
{C}
{D}
"""

g_user_prompt = """You are a medical image analysis specialist. Please follow these steps to answer the multiple-choice question:

1. Analyze the provided medical image
2. Read the question and all options thoroughly
3. Apply relevant medical knowledge and imaging interpretation principles
4. Eliminate incorrect options through systematic reasoning
5. Be extremely concise and avoid any unnecessary explanations or pleasantries.
6. Select the single best answer from choices A, B, C, or D

You think about the reasoning process as an internal monologue and then provide the final answer.
The provide ONLY the letter choice inside <answer></answer> tags.


---
IMAGE: <image>
---
QUESTION: {question}
OPTIONS:
{A}
{B}
{C}
{D}
"""


def convert_conversation_and_label(row):
    caption = row['Caption']
    answer = row['Answer']

    question = row['Question']
    A = row['Choice A']
    B = row['Choice B']
    C = row['Choice C']
    D = row['Choice D']

    tmp_prompt = g_user_prompt.format(question=question, A=A, B=B, C=C, D=D)
    conversation = []
    # add system prompt
    conversation.append(dict(role="system", content="You are a helpful assistant."))
    conversation.append(dict(role="user", content=tmp_prompt))

    label = json.dumps(dict(answer=answer, caption=caption, question=question, A=A, B=B, C=C, D=D))
    return conversation, label


def csv_to_jsonl(input_file, output_file, image_base_dir):
    if not os.path.exists(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    img_bytes_dict = {}
    jsonlfile = open(output_file, 'w', encoding='utf-8')
    csvfile = open(input_file, newline='', encoding='utf-8')
    reader = csv.DictReader(csvfile)

    for row in reader:
        image_path = os.path.join(image_base_dir, row['Figure_path'])
        img = Image.open(image_path).convert("RGB")
        width, height = img.size

        conversation, label = convert_conversation_and_label(row)
        imgs_names = [dict(
            image_path=row['Figure_path'],
            width=width,
            height=height,
        )]
        one_line = dict(
            conversations=conversation,
            label=label,
            images=imgs_names,
        )
        jsonlfile.write(json.dumps(one_line, ensure_ascii=True) + "\n")

        with open(image_path, 'rb') as f:
            img_bytes_dict[row['Figure_path']] = f.read().decode('latin1')

    jsonlfile.close()
    csvfile.close()
    print(f"Successfully converted {input_file} to {output_file}")
    return img_bytes_dict


def get_args():
    parser = argparse.ArgumentParser(description="CSV to JSONL Converter")
    parser.add_argument(
        "--csv_inputs", type=str, required=True, nargs='*', help="Path to input CSV files"
    )
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    img_bytes_dict = {}

    for csv_input in args.csv_inputs:
        output_filename = os.path.join(args.output_dir, os.path.basename(csv_input) + ".jsonl")
        img_bytes_dict.update(csv_to_jsonl(csv_input, output_filename, args.image_dir))

    lmdb_env = lmdb.open(os.path.join(args.output_dir, "img_file.lmdb"), map_size=10 * 2**40)
    with lmdb_env.begin(write=True) as txn:
        for key, value in img_bytes_dict.items():
            res = txn.put(key.encode(), value.encode('latin1'))
            assert res, "write to the lmdb should be succ"


if __name__ == "__main__":
    main()
