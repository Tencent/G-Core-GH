# DSV4 official ↔ SGLang FP8 checkpoint conversion

Convert between:

| | Path |
|---|---|
| Official | `deepseek-ai/DeepSeek-V4-Flash` (FP4 experts, E8M0 scales, FP8 `wo_a`) |
| SGLang | `sgl-project/DeepSeek-V4-Flash-FP8` (FP8 experts, float32 scales, BF16 `wo_a`) |

## What changes

### `official2sgl`
- **MoE experts**: int8-packed FP4 → FP8 via DeepSeek `cast_e2m1fn_to_e4m3fn` (lossless rebase); scale becomes float32 128×128
- **Dense FP8**: weight bytes unchanged; `float8_e8m0fnu` scale → `float32`
- **`attn.wo_a`**: FP8+scale → BF16 (no `.scale` written; avoids the published SGL phantom-key bug)
- **config**: drops `expert_dtype`

### `sgl2official`
- **MoE experts**: FP8 → requantize to FP4 packed + E8M0 (1×32). Not bit-exact vs original official FP4, but within FP4 noise after dequant
- **Dense FP8**: weight bytes unchanged; float32 scale → `float8_e8m0fnu`
- **`attn.wo_a`**: BF16 → FP8 + E8M0
- **config**: sets `expert_dtype: "fp4"`

## Multi-GPU execution

Both `convert.py` and `compare_hf_ckpts.py` are launched with **`mpirun`, one rank per
GPU**. The heavy quant/dequant math runs on the GPU (`--device cuda`, default); the
work is split deterministically with no cross-rank tensor communication — only small
metadata is gathered to rank 0 at the end.

- **`convert.py`**: shards are partitioned across ranks by **file size** (greedy
  bin-packing) for balanced load. Each rank reads / converts / writes its own shard
  files (disjoint, no write conflicts). Rank 0 copies aux files + config, then gathers
  each rank's weight-map fragment to write the single `model.safetensors.index.json`.
- **`compare_hf_ckpts.py`**: shared keys are split **round-robin** across ranks; each
  rank dequantizes + compares its slice on-GPU; rank 0 broadcasts the key coverage and
  gathers per-tensor stats.

Set `--device cpu` (or launch without `mpirun` → single rank) to fall back to CPU.
`mpi4py` is required for multi-rank; without it the scripts run as a single process.

## Usage

From `gcore-dev` root. The wrapper scripts hold the src/dst paths and the cluster
`mpirun` flags — **edit the variables inside the script**, then run:

```bash
# Convert (edit DIRECTION / SRC / DST inside the script)
bash tools/dsv4_quant_and_dequant/scripts/run_convert.sh

# Compare converted vs published (edit --a / --b inside the script)
bash tools/dsv4_quant_and_dequant/scripts/run_check_hf_ckpts.sh
```

Or invoke directly (single node, all visible GPUs):

```bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1   # one compute thread per rank; GPU does the work

mpirun -np "$(python3 -c 'import torch;print(torch.cuda.device_count())')" \
  python3 -u tools/dsv4_quant_and_dequant/convert.py \
  --direction official2sgl \
  --src /path/to/official \
  --dst /path/to/out_sgl_fp8 \
  --device cuda
```

> Launch with `-np <num_gpus>`. Using more ranks than GPUs makes ranks share cards
> (`local_rank % device_count`) → contention/OOM; using more ranks than
> shards/keys just leaves the extra ranks idle (still correct output).

## Verify

**CPU vs CUDA bit-exactness** of the kernels (run once per new torch/GPU build; confirms
the `float8_e8m0fnu` casts and quant/dequant paths match CPU byte-for-byte):

```bash
PYTHONPATH=. python3 tools/dsv4_quant_and_dequant/test_cuda_parity.py
```

**Checkpoint value comparison** (dtype/shape coverage + per-tensor dequantized diff):

```bash
mpirun -np 8 python3 tools/dsv4_quant_and_dequant/compare_hf_ckpts.py \
  --a test_convert/official2sgl_full/ \
  --b hf-hub/sgl-project/DeepSeek-V4-Flash-FP8 \
  --a-name converted --b-name published_sgl --device cuda
```

Dense weights should be byte-equal; experts should match after dequant even if raw FP8
codes differ. Coverage differences (`only_A/only_B`) and phantom index entries are
reported and factored into the final `VERDICT`.

### Note on the published SGL checkpoint

Two harmless quirks exist in `sgl-project/DeepSeek-V4-Flash-FP8` (our output does *not*
reproduce them, and they don't affect loading):

- `model.safetensors.index.json` `total_size` is under-reported (~159 GB) — it carries
  the *official* FP4-packed size, while the real FP8 data is ~294 GB. Our converted
  index reports the correct 294 GB.
- 44 phantom `layers.*.attn.wo_a.scale` (+ `mtp.*`) keys are listed in the index but the
  tensors are absent from the shards (SGL `wo_a` has no real scale). `compare_hf_ckpts.py`
  treats these as phantom entries.

The `sgl2official` output, by contrast, matches the published official checkpoint
exactly (identical keys, dtypes, shapes, and `total_size`).
