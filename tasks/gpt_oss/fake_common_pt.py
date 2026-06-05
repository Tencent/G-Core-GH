import torch
import argparse
import os
from argparse import Namespace
from enum import Enum


def main():
    parser = argparse.ArgumentParser(description='Generate fake common.pt file')
    parser.add_argument(
        '--mlm-path', type=str, required=True, help='Path where to save the common.pt file'
    )

    args = parser.parse_args()

    # Create the directory if it doesn't exist
    os.makedirs(args.mlm_path, exist_ok=True)

    # Create the fake common.pt content
    fake_common_data = {
        'args':
            Namespace(
                pipeline_model_parallel_size=1,
                tensor_model_parallel_size=1,
                train_data_consuming_progresses={}
            ),
        'checkpoint_version':
            3.0,
        'iteration':
            1,
        'num_floating_point_operations_so_far':
            0
    }

    # Save to common.pt
    common_pt_path = os.path.join(args.mlm_path, 'common.pt')
    torch.save(fake_common_data, common_pt_path)

    print(f"Successfully created fake common.pt at: {common_pt_path}")


if __name__ == "__main__":
    main()
