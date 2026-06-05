import sys
import os
import pandas as pd

# data source path and target path
parquet_data_dir = sys.argv[1]
target_dir = sys.argv[2]

if not os.path.exists(target_dir):
    os.mkdir(target_dir)

for file_name in os.listdir(parquet_data_dir):
    file_path = os.path.join(parquet_data_dir, file_name)
    if file_name.startswith("train"):
        save_dir = os.path.join(target_dir, "train")
    else:
        save_dir = os.path.join(target_dir, "eval")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, file_name.replace(".parquet", ".jsonl"))

    df = pd.read_parquet(file_path)
    df.to_json(save_path, orient='records', lines=True)
    print(f"save {save_path}")
