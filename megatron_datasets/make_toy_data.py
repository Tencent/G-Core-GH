import sys
import os
import json
import time
import string
import random


def make(dname):
    os.makedirs(dname, exist_ok=True)
    fname = os.path.join(dname, 'part-1.jsonl')
    with open(fname, 'w') as outf:
        for i in range(10000):
            o = {
                'input': ''.join(random.choices(string.ascii_letters, k=1024)),
                'target': ''.join(random.choices(string.ascii_letters, k=512)),
                'instruction': ''.join(random.choices(string.ascii_letters, k=1024)),
                'chosen': ''.join(random.choices(string.ascii_letters, k=512)),
                'rejected': ''.join(random.choices(string.ascii_letters, k=512)),
            }
            outf.write(json.dumps(o) + '\n')


make('toy-data/pretrain/baike')
make('toy-data/pretrain/arxiv')
make('toy-data/sft')
make('toy-data/rm')
make('toy-data/ppo')
