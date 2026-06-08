RCDIR="/work/wepsdl"
# NB: gcore-dev/tests/test_gpatch_v4 must be on PYTHONPATH so that ray
# workers (started via mpirun-init-ray.sh below, which forwards
# PYTHONPATH via `-x PYTHONPATH`) can import the @ray.remote worker
# functions defined inline in those test files. cloudpickle ships them
# by reference using the bare module name (e.g.
# `test_qwen3_6_moe_text_only_sft`), and that dir has no __init__.py,
# so the worker can only resolve the import if the dir is on sys.path.
# NB: Ray must be initialised *before* test_gfused so that ray workers
# inherit the full PYTHONPATH (including Megatron-LM) forwarded via
# `-x PYTHONPATH` in mpirun-init-ray.sh.  Without this,
# test_gfused/test_gated_delta_net.py connects to a stale cluster whose
# workers have no Megatron-LM on sys.path and fail with
# "ModuleNotFoundError: No module named 'megatron'".
export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/gcore-dev/tests:$RCDIR/gcore-dev/tests/test_gpatch_v4:$RCDIR/Megatron-LM:$RCDIR/mbridge:$RCDIR/Megatron-Bridge/src:$PYTHONPATH"

source tests/test_gpatch_v4/mpirun-stop-ray.sh
source tests/test_gpatch_v4/mpirun-init-ray.sh

pytest -v -s --timeout=1800 tests/test_gfused

pytest -v -s --timeout=1800 tests/test_gpatch_v4
