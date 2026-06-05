#!/bin/bash
# Convert or test the Qwen3.6-35B-A3B FP8 checkpoint.
#
# Examples:
#   # Convert BF16 -> FP8 with official-style skips and MTP post-processing.
#   bash tools/convert_fp8/qwen3_5/scripts/run_quantize_qwen3_6_moe.sh --mode convert
#
#   # Compare FP8 generate against BF16.
#   bash tools/convert_fp8/qwen3_5/scripts/run_quantize_qwen3_6_moe.sh --mode test
#
#   # Pass extra args to the underlying Python script after "--".
#   bash tools/convert_fp8/qwen3_5/scripts/run_quantize_qwen3_6_moe.sh --mode test -- --prompt "What is FP8?"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
QWEN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONVERT_FP8_DIR="$(cd "${QWEN_DIR}/.." && pwd)"
LOG_DIR="${CONVERT_FP8_DIR}/logs"
mkdir -p "$LOG_DIR"

MODE="convert"
INPUT_HF_PATH="/work/wepsdl/gcore-dev/hf-hub/Qwen/Qwen3.6-35B-A3B"
OUTPUT_FP8_PATH="/work/wepsdl/gcore-dev/Qwen3.6-35B-A3B-FP8-official-skip"
BF16_PATH="$INPUT_HF_PATH"
FP8_PATH="$OUTPUT_FP8_PATH"
MAX_NEW_TOKENS=512
EXTRA_ARGS=()

usage() {
    cat <<EOF
Usage:
  $0 --mode convert [options] [-- extra_quantize_args...]
  $0 --mode test    [options] [-- extra_generate_args...]

Modes:
  convert    Convert BF16 HF checkpoint to FP8.
  test       Run HuggingFace generate and compare FP8 vs BF16.

Options:
  -m, --mode MODE              convert | test (default: convert)
  --input-hf-path PATH         BF16 input path for convert
  --output-fp8-path PATH       FP8 output path for convert
  --bf16-path PATH             BF16 path for test
  --fp8-path PATH              FP8 path for test
  --max-new-tokens N           Test generation max tokens (default: 512)
  -h, --help                   Show this help

Defaults:
  BF16: $INPUT_HF_PATH
  FP8:  $OUTPUT_FP8_PATH
EOF
}

require_value() {
    local flag="$1"
    if [[ $# -lt 2 || -z "${2:-}" ]]; then
        echo "Missing value for $flag" >&2
        usage >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        convert|quantize)
            MODE="convert"
            shift
            ;;
        test|generate)
            MODE="test"
            shift
            ;;
        -m|--mode)
            require_value "$1" "${2:-}"
            MODE="$2"
            shift 2
            ;;
        --input-hf-path|--input)
            require_value "$1" "${2:-}"
            INPUT_HF_PATH="$2"
            BF16_PATH="$2"
            shift 2
            ;;
        --output-fp8-path|--output)
            require_value "$1" "${2:-}"
            OUTPUT_FP8_PATH="$2"
            FP8_PATH="$2"
            shift 2
            ;;
        --bf16-path)
            require_value "$1" "${2:-}"
            BF16_PATH="$2"
            shift 2
            ;;
        --fp8-path)
            require_value "$1" "${2:-}"
            FP8_PATH="$2"
            shift 2
            ;;
        --max-new-tokens)
            require_value "$1" "${2:-}"
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$MODE" in
    convert)
        LOG_FILE="${LOG_DIR}/quantize_qwen3.6_35b_fp8_$(date +%Y%m%d_%H%M%S)_$$.log"
        echo "=== Offline FP8 Quantization (FineGrainedFP8Config) ==="
        echo "Input:  $INPUT_HF_PATH"
        echo "Output: $OUTPUT_FP8_PATH"
        echo "Log:    $LOG_FILE"
        echo ""

        set +e
        python "$QWEN_DIR/quantize_hf_to_fp8.py" \
            --input-hf-path "$INPUT_HF_PATH" \
            --output-fp8-path "$OUTPUT_FP8_PATH" \
            --skip-policy official \
            --block-size 128 128 \
            --activation-scheme dynamic \
            "${EXTRA_ARGS[@]}" \
            > "$LOG_FILE" 2>&1
        status=$?
        set -e
        echo "EXIT_CODE=$status" >> "$LOG_FILE"
        echo "LOG=$LOG_FILE"
        echo "EXIT_CODE=$status"
        exit "$status"
        ;;
    test)
        LOG_FILE="${LOG_DIR}/compare_bf16_fp8_generate_$(date +%Y%m%d_%H%M%S)_$$.log"
        echo "=== HuggingFace Generate Test (FP8 vs BF16) ==="
        echo "FP8:    $FP8_PATH"
        echo "BF16:   $BF16_PATH"
        echo "Log:    $LOG_FILE"
        echo ""

        set +e
        python "$QWEN_DIR/tests/test_fp8_hf_generate.py" \
            --fp8-path "$FP8_PATH" \
            --bf16-path "$BF16_PATH" \
            --max-new-tokens "$MAX_NEW_TOKENS" \
            "${EXTRA_ARGS[@]}" \
            > "$LOG_FILE" 2>&1
        status=$?
        set -e
        echo "EXIT_CODE=$status" >> "$LOG_FILE"
        echo "LOG=$LOG_FILE"
        echo "EXIT_CODE=$status"
        exit "$status"
        ;;
    *)
        echo "Unknown mode: $MODE" >&2
        usage >&2
        exit 2
        ;;
esac
