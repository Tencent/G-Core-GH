RCDIR=$PWD

export PYTHONPATH="$RCDIR/tools/convert_fp8/qwen3_5:$PYTHONPATH"


pytest -v -s --timeout=1800 tools/convert_fp8/qwen3_5/tests/test_qwen3_official_fp8_skips.py
pytest -v -s --timeout=1800 tools/convert_fp8/qwen3_5/tests/test_qwen3_mtp_fp8_postprocess.py
