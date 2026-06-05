import os
import argparse
import json


def get_args():
    parser = argparse.ArgumentParser(description="Build the GDatasetV4 Meta Json File")
    parser.add_argument(
        "--json_inputs", type=str, required=True, nargs='*', help="to input json files"
    )
    parser.add_argument("--output_fullpath", type=str, required=True, help="Output json file path")
    parser.add_argument("--description", type=str, default="", help="dataset description")
    parser.add_argument("--name", type=str, default="", help="dataset name")
    parser.add_argument(
        '--rebuild', action="store_true", help="rebuild whether the Meta file exists"
    )
    parser.add_argument("--lmdb_port", type=int, default=None, help="lmdb port")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    if os.path.exists(args.output_fullpath) and not args.rebuild:
        print(
            f"the file already exists, it will not be rebuilt. If you want to rebuild it, add the option --rebuild"
        )
        return

    assert len(args.json_inputs) > 0
    data_files = []
    data_file_num_lines = []
    for json_input in args.json_inputs:
        assert os.path.exists(json_input)
        data_files.append(dict(fp=json_input))
        with open(json_input, "r") as f:
            data_file_num_lines.append(len(f.readlines()))

    meta_dict = dict(
        name=args.name,
        description=args.description,
        data_files=data_files,
        data_file_num_lines=data_file_num_lines,
    )
    if args.lmdb_port is not None:
        meta_dict["lmdb_http_url"] = f"http://127.0.0.1:{args.lmdb_port}/read"

    with open(args.output_fullpath, "w") as f:
        json.dump(meta_dict, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
