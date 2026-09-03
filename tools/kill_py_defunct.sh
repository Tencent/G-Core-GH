#!/bin/bash
# kill 所有的 defunct 进程，循环 n 轮（默认 10）

n=${1:-10}

for ((i=1; i<=n; i++)); do
  pkill -9 -f python
  echo "=== Round $i/$n ==="
  pids=$(ps -eo pid,stat | awk '$2 ~ /Z/ {print $1}')

  if [ -z "$pids" ]; then
    echo "No defunct processes found"
    continue
  fi

  echo "Found defunct PIDs: $pids"
  for pid in $pids; do
    kill -9 "$pid" 2>/dev/null || true
  done
done
