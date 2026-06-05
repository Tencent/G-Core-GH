import argparse
import torch
from pathlib import Path


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='Remove train_data_consuming_progresses from checkpoint'
    )
    parser.add_argument(
        '--mlm-path', type=str, required=True, help='Path to the directory containing common.pt'
    )
    args = parser.parse_args()

    # 构建 common.pt 的完整路径
    mlm_path = Path(args.mlm_path)
    common_pt_path = mlm_path / 'common.pt'

    # 检查文件是否存在
    if not common_pt_path.exists():
        print(f"Error: File not found: {common_pt_path}")
        return

    print(f"Loading checkpoint from: {common_pt_path}")

    # 读取 common.pt 文件
    checkpoint_dict = torch.load(common_pt_path, map_location='cpu', weights_only=False)

    backup_path = common_pt_path.with_suffix('.pt.backup')
    print(f"Creating backup at: {backup_path}")
    torch.save(checkpoint_dict, backup_path)

    # 检查是否包含 args
    if 'args' not in checkpoint_dict:
        print("Error: 'args' key not found in checkpoint dictionary")
        print(f"Available keys: {list(checkpoint_dict.keys())}")
        return

    args_namespace = checkpoint_dict['args']

    # 检查 args 是否是 Namespace 对象
    if not isinstance(args_namespace, argparse.Namespace):
        print(f"Warning: 'args' is not a Namespace object, it's a {type(args_namespace)}")

    if hasattr(args_namespace, 'exit_signal'):
        print(f"Found 'exit_signal' attribute")
        delattr(args_namespace, 'exit_signal')
        print("Successfully deleted 'exit_signal' attribute")

    # 检查并删除 train_data_consuming_progresses 属性
    if hasattr(args_namespace, 'train_data_consuming_progresses'):
        print(f"Found 'train_data_consuming_progresses' attribute")
        delattr(args_namespace, 'train_data_consuming_progresses')
        print("Successfully deleted 'train_data_consuming_progresses' attribute")

    print(f"Saving modified checkpoint to: {common_pt_path}")
    torch.save(checkpoint_dict, common_pt_path)
    print("Done!")


if __name__ == '__main__':
    main()
