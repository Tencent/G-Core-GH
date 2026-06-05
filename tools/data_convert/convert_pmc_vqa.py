import os
import csv
import json
import argparse

from PIL import Image
import lmdb


def csv_to_jsonl(input_file, output_file, image_base_dir):
    if not os.path.exists(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    img_bytes_dict = {}
    with open(output_file, 'w', encoding='utf-8') as jsonlfile:
        with open(input_file, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                image_path = os.path.join(image_base_dir, row['Figure_path'])
                img = Image.open(image_path).convert("RGB")
                width, height = img.size
                question = row['Question']
                answer = row['Answer']
                choices = [row[f'Choice {x}'] for x in ("A", "B", "C", "D")]
                choice_str = " , ".join(choices)

                result = {
                    "conversations":
                        [
                            {
                                "role":
                                    "user",
                                "content":
                                    f"You are a medical expert, please observe the following picture and answer this question accurately: {question} Choose from the following options and response with only the letter option: {choice_str} A letter of A/B/C/D is all you need to return and absolutely nothing else. <image>"
                            }, {
                                "role": "assistant",
                                "content": answer
                            }
                        ],
                    "images":
                        [{
                            "image_path": row['Figure_path'],
                            "width": width,
                            "height": height,
                        }]
                }
                with open(image_path, 'rb') as f:
                    img_bytes_dict[row['Figure_path']] = f.read().decode('latin1')

                json.dump(result, jsonlfile, ensure_ascii=False)
                jsonlfile.write('\n')
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
