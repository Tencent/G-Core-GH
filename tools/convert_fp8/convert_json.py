import argparse
import os
import json

parser = argparse.ArgumentParser(
    prog='convert_json.py', description='Convert JSON checkpoint from FP8 to FP16'
)

parser.add_argument('--remove_fp8', action='store_true', help='Convert FP8 to FP16')
parser.add_argument('--remove_mtp', action='store_true', help='Remove MTP from json')
parser.add_argument('--load_hf_dir', required=True, help='Input hf_dir')
parser.add_argument('--save_hf_dir', required=True, help='Output hf_dir')


def convert_config_json(config_json, remove_fp8=False, remove_mtp=False):
    if remove_fp8:
        config_json.pop("quantization_config", None)
    if remove_mtp:
        config_json.pop("num_nextn_predict_layers", None)
    return config_json


def convert_index_json(index_json, remove_fp8=False, remove_mtp=False):
    new_index_json = {"metadata": index_json["metadata"]}
    new_weight_map = {}
    for key, fname in index_json["weight_map"].items():
        if remove_fp8 and key.endswith(".weight_scale_inv"):
            # print(f"key: {key} skip")
            continue
        if remove_mtp and key.startswith("model.layers.61."):
            continue
        new_weight_map[key] = fname
    new_index_json["weight_map"] = new_weight_map
    return new_index_json


def convert_config_json_file(load_hf_dir, save_hf_dir, remove_fp8=False, remove_mtp=False):
    config_file = os.path.join(load_hf_dir, "config.json")
    assert os.path.exists(config_file), f"{config_file} does not exist"
    with open(config_file, "r") as f:
        config_json = json.load(f)
        config_json = convert_config_json(config_json, remove_fp8, remove_mtp)
        new_config_file = os.path.join(save_hf_dir, "config.json")
        with open(new_config_file, "w") as new_f:
            json.dump(config_json, new_f, indent=2)
    print(f"Converted {config_file} to {new_config_file} done.")


def convert_index_json_file(load_hf_dir, save_hf_dir, remove_fp8=False, remove_mtp=False) -> None:
    index_file = os.path.join(load_hf_dir, "model.safetensors.index.json")
    assert os.path.exists(index_file), f"{index_file} does not exist"
    with open(index_file, "r") as f:
        index_json = json.load(f)
        new_index_json = convert_index_json(index_json, remove_fp8, remove_mtp)
        new_index_file = os.path.join(save_hf_dir, "model.safetensors.index.json")
        with open(new_index_file, "w") as new_f:
            json.dump(new_index_json, new_f, indent=2)
    print(f"Converted {index_file} to {new_index_file} done.")


def main():
    args = parser.parse_args()
    convert_config_json_file(args.load_hf_dir, args.save_hf_dir, args.remove_fp8, args.remove_mtp)
    convert_index_json_file(args.load_hf_dir, args.save_hf_dir, args.remove_fp8, args.remove_mtp)


if __name__ == "__main__":
    main()
