# MoE Offline Routing-Map Permutation

Offline expert-id permutation for Qwen3.6-MoE HF checkpoints.

For each MoE layer, computes an EP-agnostic permutation of the 256 routed
experts via recursive balanced bisection (8 levels of `balanced_packing(num_packs=2)`).
The resulting permutation makes any `EP_size = 2^k` (k = 1..8) contiguous
segmentation of the new expert order load-balanced.

## Permutation convention

Throughout this tool we use **position-major** permutation:

- `perm[L, p] = old_expert_id` — "at new position p of layer L sits the
  expert that originally had id `perm[L, p]`".
- The HF tensor rewrite is therefore a **gather** along dim 0 with `perm[L]`:
  ```
  new_gate[L]    = old_gate[L][perm[L]]            # [256, 2048]
  new_up[L]      = old_up[L][perm[L]]              # [256, 1024, 2048]
  new_down[L]    = old_down[L][perm[L]]            # [256, 2048, 512]
  ```
- `inv_perm[L, q] = new_position_of_old_expert_q`. Useful for "where did
  old expert q go?" queries; not needed by the rewrite itself.

## Scope (Qwen3.6-35B-A3B specifics)

- 40 routed-MoE layers, each with 256 experts, top-8 routing.
- Permuted tensors per layer (only these 3, dim 0 is expert id):
  - `model.language_model.layers.{L}.mlp.gate.weight`            `[256, 2048]`
  - `model.language_model.layers.{L}.mlp.experts.gate_up_proj`   `[256, 1024, 2048]`
  - `model.language_model.layers.{L}.mlp.experts.down_proj`      `[256, 2048, 512]`
- Untouched: `mlp.shared_expert.*`, `mlp.shared_expert_gate.*`, all attention
  tensors, embeddings, lm_head.
- **MTP block (`mtp.layers.0.mlp.*`) is left unchanged** — we have no routing
  statistic for it; permuting its routed experts without data would be arbitrary.
- Qwen3.6-MoE has no `e_score_correction_bias` / router bias term (verified by
  scanning the safetensors index).

## Pipeline

```
step_1_pp{0..3}.jsonl    # 64K dump, recompute_factor==1 records
        |
        v
build_routing_map.py     # aggregate -> [40, 256] weight -> recursive bisect
        |
        v
routing_map.json         # perm, inv_perm, weights, multi-EP imbalance
        |
        v
repermute_hf_ckpt.py     # rewrite 26 safetensors shards
        |
        v
Qwen3.6-35B-A3B-permuted/   # rewritten checkpoint (side-by-side)
        |
        v
verify_equivalence.py    # per-layer manual MoE forward equivalence
        |                 # + optional full HF forward cos_sim check
        v
viz_repermute.py         # before/after heatmaps + repermute_report.md
```

The four pipeline steps above can be driven either individually (CLI on each
script) or end-to-end via the Python orchestrator `repermute_pipeline.py`
(see `run.sh` for the canonical invocation).

### Standalone analysis tool (not part of the pipeline)

`viz_debug_router.py` is a **separate, manually invoked** tool. It is NOT
called by `repermute_pipeline.py` or `run.sh`. It consumes a
`verify_equivalence` log that happens to contain
`DEBUG router select router_indices=tensor(...)` dumps (only produced when
the underlying transformers/sglang router has debug prints enabled), and
emits per-layer hot top-k tables plus src-vs-(dst->old) overlap plots.
Invoke only when you want that specific post-hoc diagnosis.

## End-to-end runner

```bash
bash tools/moe_offline_repermute/run.sh
```

or equivalently:

```bash
python -m tools.moe_offline_repermute.repermute_pipeline \
  --source-dump  /path/to/moe_dist_64K \
  --src-ckpt     hf-hub/Qwen/Qwen3.6-35B-A3B \
  --dst-ckpt     /path/to/Qwen3.6-35B-A3B-permuted \
  --work-dir     /path/to/moe_repermute_work
```

The orchestrator chains `build_routing_map` -> `repermute_hf_ckpt` (dry-run
then rewrite) -> `verify_equivalence` -> `visualize_repermute`. Model shape
(`num_hidden_layers`, `num_experts`) is resolved via
`transformers.AutoConfig.from_pretrained(src_ckpt).get_text_config()`, so VL
checkpoints whose MoE fields live under `text_config` (e.g. Qwen3.6-VL) and
flat non-VL MoE configs are both handled uniformly. Pass `--num-layers` /
`--num-experts` to override.

## Files

- `eplb_vendored.py`        — vendored `balanced_packing` and `inverse` from
  `EPLB_visualization/eplb_np.py` (no upstream package).
- `aggregate.py`            — jsonl -> `[L, 256]` weight aggregation.
- `perm.py`                 — recursive balanced bisection + helpers.
- `build_routing_map.py`    — step 1: aggregate + compute perm.
- `repermute_hf_ckpt.py`    — step 2: rewrite safetensors shards.
- `verify_equivalence.py`   — step 3: bit-exact per-layer check + optional
                              full HF forward `cos_similarity` check.
- `viz_repermute.py`        — step 4: EP load heatmaps + placement grids +
                              `repermute_report.md`.
- `viz_debug_router.py`     — **standalone, not in pipeline**: consume a
                              `verify_equivalence` log containing
                              `DEBUG router select router_indices=tensor(...)`
                              blocks; emit per-layer hot top-k tables and
                              src-vs-(dst->old) overlap plots.
- `repermute_pipeline.py`   — Python orchestrator chaining the steps above.
- `run.sh`                  — convenience wrapper that invokes the
                              orchestrator with typical paths.
