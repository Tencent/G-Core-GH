

PYTHONPATH="$PWD:/work/wepsdl/gcore-dev:$PYTHONPATH"


python3 -m tools.moe_offline_repermute.repermute_pipeline \
  --source-dump /mnt/ceph-hz1-csp/mm-base-plt2/user_xiaotaoliu/ft_local/moe_dist_64K/ \
  --src-ckpt hf-hub/Qwen/Qwen3.6-35B-A3B \
  --dst-ckpt /mnt/ceph-hz1-csp/mm-base-plt2/user_xiaotaoliu/project/workspace3/test/Qwen3.6-35B-A3B-permuted \
  --work-dir /mnt/ceph-hz1-csp/mm-base-plt2/user_xiaotaoliu/project/workspace3/test/moe_repermute_work
