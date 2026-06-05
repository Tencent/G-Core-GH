import argparse
from typing import List

import lmdb
from fastapi import FastAPI
import uvicorn
import asyncio

app = FastAPI()
txn = None


@app.post("/read")
async def read_data(keys: List[str]):
    global txn

    result = []
    for key in keys:
        loop = asyncio.get_event_loop()
        value = await loop.run_in_executor(None, lambda: txn.get(key.encode()))
        if value:
            result.append({"key": key, "value": value.decode('latin1'), "res": True})
        else:
            result.append({"key": key, "value": "key not found", "res": False})

    return {"data": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='lmdb server')
    group.add_argument('--lmdb-path', type=str, required=True, help="lmdb文件的路径")
    group.add_argument('--lmdb-map-size', type=int, required=True, help="lmdb内存缓冲区大小，单位为GB")
    group.add_argument('--lmdb-port', type=int, required=True, help="lmdb服务端口")
    args = parser.parse_args()
    print(f"LMDB run args:{args}")

    # 创建 LMDB 环境
    env = lmdb.open(
        args.lmdb_path,
        map_size=args.lmdb_map_size * 1024 * 1024 * 1024,
        readonly=True,
        lock=False,
    )
    txn = env.begin(write=False)

    uvicorn.run(app, host="127.0.0.1", port=args.lmdb_port, log_level="error")
