"""
Augment pt files: randomly select 2 out of 8 pt files,
for each sample, randomly repeat `tokens` by 3~12x and update `sequence_lengths`.

Usage:
    python augment_pt.py --input_dir <dir_containing_pt_files> [--output_dir <output_dir>]

If --output_dir is not specified, files are overwritten in place.
"""

import argparse
import glob
import os
import random
import shutil
import torch


def augment_file(filepath, output_path):
    """Load a pt file (List[Dict[str, List]]), augment tokens, save."""
    data = torch.load(filepath, map_location="cpu")
    assert isinstance(data, list), f"Expected list, got {type(data)}"

    for batch_idx, batch in enumerate(data):
        assert isinstance(batch, dict), f"Expected dict, got {type(batch)}"
        tokens_list = batch["tokens"]
        seqlen_list = batch["sequence_lengths"]
        num_samples = len(tokens_list)

        for i in range(num_samples):
            token = tokens_list[i]
            factor = random.randint(3, 12)

            if torch.is_tensor(token):
                # Repeat 1D tensor
                new_token = token.repeat(factor)
            elif isinstance(token, list):
                new_token = token * factor
            else:
                raise TypeError(f"Unexpected tokens type: {type(token)}")

            tokens_list[i] = new_token

            # Update sequence_lengths
            new_len = len(new_token)
            if torch.is_tensor(seqlen_list[i]):
                seqlen_list[i] = torch.tensor(new_len, dtype=seqlen_list[i].dtype)
            else:
                seqlen_list[i] = new_len

            if i < 3:
                orig_len = len(token)
                print(
                    f"  batch[{batch_idx}] sample[{i}]: factor={factor}, "
                    f"orig_len={orig_len}, new_len={new_len}"
                )

    torch.save(data, output_path)
    print(f"Saved augmented file to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Augment pt files by repeating tokens")
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Directory containing origin_data_*.pt files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: overwrite in place)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="origin_data_*.pt",
        help="Glob pattern for pt files (default: origin_data_*.pt)"
    )
    parser.add_argument(
        "--num_select", type=int, default=2, help="Number of files to randomly select (default: 2)"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # Find all matching pt files
    pt_files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    print(f"Found {len(pt_files)} pt files: {[os.path.basename(f) for f in pt_files]}")

    assert len(pt_files) >= args.num_select, \
        f"Need at least {args.num_select} files, but found {len(pt_files)}"

    # Randomly select files
    selected_files = random.sample(pt_files, args.num_select)
    print(
        f"Randomly selected {args.num_select} files: {[os.path.basename(f) for f in selected_files]}"
    )

    output_dir = args.output_dir or args.input_dir
    os.makedirs(output_dir, exist_ok=True)

    selected_set = set(selected_files)

    # Copy unselected files as-is to output_dir
    for filepath in pt_files:
        if filepath not in selected_set:
            basename = os.path.basename(filepath)
            output_path = os.path.join(output_dir, basename)
            if os.path.abspath(filepath) != os.path.abspath(output_path):
                shutil.copy2(filepath, output_path)
                print(f"Copied (unmodified): {basename}")
            else:
                print(f"Skipped (same path): {basename}")

    # Augment selected files
    for filepath in selected_files:
        basename = os.path.basename(filepath)
        output_path = os.path.join(output_dir, basename)
        print(f"\nProcessing (augmenting): {basename}")
        augment_file(filepath, output_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
