import argparse
from pathlib import Path

import yaml
from megatron.energon.flavors import CrudeJsonlDatasetFactory


def build_energon_yaml(data_list, yaml_path):
    meta = {}
    meta["__module__"] = "megatron.energon",
    meta["__class__"] = "MetadatasetV2",
    meta["splits"] = {}
    meta["splits"]["train"] = []
    train_split = meta["splits"]["train"]
    for e in datas_list:
        train_split.append({"path": f"{e}"})
    with open(yaml_path, "w") as f:
        yaml.dump(meta, f)


def main(args):
    data_path = Path(args.path)
    path_list = []
    if data_path.is_dir():
        files = data_path.glob("./*.jsonl")
        assert len(files) > 0, f"no jsonl data in dir {data_path}"
    else:
        assert ".jsonl" == data_path.suffix, "only support jsonl for now"
        path_list = [data_path]

    for path in path_list:
        CrudeJsonlDatasetFactory.prepare_dataset(path)
    if len(path_list) > 1:
        build_energon_yaml(path_list, data_path / "energon_meta.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="prepare args")
    parser.add_argument("--path", type=str, required=True, help="HuggingFace model path")
    args = parser.parse_args()
    main(args)
