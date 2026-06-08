'''
upload checkpoint and data to **private repo** for testing purpose
for token visit https://mirrors.tencent.com/#/private/generic2/detail?repo_name=wepsdlpriv

```bash
my_token=xxx
python3 tests/upload.py --username nrwu --token $my_token --src hf-hub/wechat/data
python3 tests/upload.py --username nrwu --token $my_token --src hf-hub/wechat/oteam4_3/vae
python3 tests/upload.py --username nrwu --token $my_token --src hf-hub/wechat/oteam4_4-step-10000
python3 tests/upload.py --username nrwu --token $my_token --src hf-hub/xinhangleng/models/oteam_models/oteam44/1024/sft/1112_sft1024_cluster30w_text10w_human5w_geneval2w_seed4infer_lr1e-5_gb128/step-60000
python3 tests/upload.py --username nrwu --token $my_token --src hf-hub/wechat/welm_v4_instruct
python3 tests/upload.py --username nrwu --token $my_token --src hf-hub/wechat/welm_v4_80A3B_128k_20251015
python3 tests/upload.py --username nrwu --token $my_token --src hf-hub/wechat/oriontian_0407_part1_2_mix_sft_hf/epoch_008_step_0000240
```
'''

import argparse
import json
import os
import os.path
import pprint
import subprocess
import sys
import math
import shutil
import pprint

import socket
import psutil
import requests


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--username', type=str, required=True)
    parser.add_argument('--token', type=str, required=True)
    parser.add_argument('--src', type=str, required=True)
    parser.add_argument('--dst', type=str, default='test_data')
    args = parser.parse_args()
    return args


def get_files_in_dir_ignoring_symbol_link(d):
    if os.path.islink(d):
        return []

    if not os.path.isdir(d):
        return [d]

    fpaths = []
    for fpath in os.listdir(d):
        new_fpath = os.path.join(d, fpath)
        if os.path.isdir(new_fpath):
            tmp = get_files_in_dir_ignoring_symbol_link(new_fpath)
            fpaths.extend(tmp)
        else:
            fpaths.append(new_fpath)
    return fpaths


def upload(args):
    fpaths = get_files_in_dir_ignoring_symbol_link(args.src)
    pprint.pprint(fpaths)

    for fpath in fpaths:
        url = os.path.join(
            f"https://mirrors.tencent.com/repository/generic/wepsdlpriv/test_data", fpath
        )
        print(f'uploading {fpath} to {url}')
        with open(fpath, "rb") as fp:
            response = requests.request("PUT", url, auth=(args.username, args.token), data=fp)


if __name__ == '__main__':
    args = get_args()
    upload(args)
