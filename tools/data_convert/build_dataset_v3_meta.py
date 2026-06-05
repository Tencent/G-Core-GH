import os
import argparse
import json


def get_args():
    parser = argparse.ArgumentParser(description="Build the GDatasetV4 Meta Json File")
    parser.add_argument("--dataset_dir", type=str, required=True, help="to input json files")
    parser.add_argument("--dataset_eval_dir", type=str, default=None, help="to input json files")
    parser.add_argument("--output_fullpath", type=str, required=True, help="Output json file path")
    parser.add_argument(
        '--rebuild', action="store_true", help="rebuild whether the Meta file exists"
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    if os.path.exists(args.output_fullpath) and not args.rebuild:
        print(
            f"the file already exists, it will not be rebuilt. If you want to rebuild it, add the option --rebuild"
        )
        return
    assert os.path.isdir(args.dataset_dir), f"{args=}"

    meta_dict = dict(
        train_data_infos=dict(
            sample_small=dict(
                path=args.dataset_dir,
                probability=1.0,
                sample_rate=1.0,
            )
        )
    )

    if args.dataset_eval_dir is not None:
        meta_dict.update(
            dict(eval_data_infos=dict(sample_small=dict(path=args.dataset_eval_dir, )))
        )

    with open(args.output_fullpath, "w") as f:
        json.dump(meta_dict, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
