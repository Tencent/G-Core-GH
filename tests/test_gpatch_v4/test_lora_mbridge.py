"""Integration tests for mbridge LoRA (lora & canonical_lora) with TP=1 and TP=2.

Verifies:
  1. Training completes without errors.
  2. Loss and grad_norm stay within sane bounds (no NaN / explosion).
  3. Exported HF checkpoint satisfies:  original_base + adapter_delta == merged_export.
"""

import os
import shutil
import unittest
import json

import torch
from safetensors.torch import load_file

try:
    from gpatch_v4.configs.config import FinetuneConfig
    from gpatch_v4.trainer import FinetuneTrainer
    from gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray, load_config
except ImportError:
    pass


# ============================================================
#  Shared utilities
# ============================================================

def _load_safetensors_dir(path):
    """Load all safetensor files from a directory into a single dict (CPU)."""
    state = {}
    for fname in sorted(os.listdir(path)):
        if fname.endswith(".safetensors"):
            st = load_file(os.path.join(path, fname), device="cpu")
            state.update(st)
    return state


def _load_hf_config(model_path):
    """Load HF model config.json, resolving VLM nested configs."""
    with open(os.path.join(model_path, "config.json")) as f:
        cfg = json.load(f)
    if "text_config" in cfg:
        return cfg["text_config"]
    return cfg


def _load_vision_config(model_path):
    """Load vision_config from config.json if present."""
    with open(os.path.join(model_path, "config.json")) as f:
        cfg = json.load(f)
    return cfg.get("vision_config", None)


def _get_head_dim(hf_config):
    num_attn_heads = hf_config["num_attention_heads"]
    hidden_dim = hf_config["hidden_size"]
    return hf_config.get("head_dim") or (hidden_dim // num_attn_heads)


def _parse_lora_pairs(adapter_path):
    """Load adapter and parse into {module_path: {lora_A, lora_B}} pairs."""
    adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
    with open(adapter_config_path) as f:
        adapter_cfg = json.load(f)

    adapter_state = load_file(
        os.path.join(adapter_path, "adapter_model.safetensors"), device="cpu"
    )

    rank = adapter_cfg["r"]
    alpha = adapter_cfg["lora_alpha"]
    scaling = alpha / rank

    print(f"  LoRA rank={rank}, alpha={alpha}, scaling={scaling}")
    print(f"  Adapter has {len(adapter_state)} parameters")

    lora_pairs = {}
    for key in adapter_state:
        if "lora_A.weight" in key:
            module_path = key.replace(".lora_A.weight", "")
            lora_pairs.setdefault(module_path, {})["lora_A"] = adapter_state[key]
        elif "lora_B.weight" in key:
            module_path = key.replace(".lora_B.weight", "")
            lora_pairs.setdefault(module_path, {})["lora_B"] = adapter_state[key]

    print(f"  Found {len(lora_pairs)} LoRA module pairs")
    return lora_pairs, scaling


def _load_base_as_bf16(original_path):
    """Load original HF weights and cast floating-point tensors to bf16."""
    original = _load_safetensors_dir(original_path)
    return {
        k: v.to(torch.bfloat16) if v.is_floating_point() else v
        for k, v in original.items()
    }


def _verify_and_compare(merged, reconstructed, adapted_keys=None):
    """Compare merged export vs reconstructed weights using torch.equal.

    Parameters
    ----------
    merged : dict
    reconstructed : dict
    adapted_keys : set or None
        Keys that had LoRA deltas applied. Used for EP expert verification.

    Notes
    -----
    3D expert weights (ndim==3) are skipped from the strict comparison because
    the EP distributed pipeline introduces unavoidable precision differences:
    - Non-adapted experts: base weight round-trip through EP scatter/gather
    - Adapted experts: LoRA adapter divergence across EP ranks
    For adapted 3D experts, rank 0's slice is verified separately.
    """
    # Auto-detect: if merged doesn't have mtp keys but reconstructed does, remove them
    has_mtp_in_merged = any(k.startswith("mtp.") for k in merged)
    if not has_mtp_in_merged:
        mtp_keys = [k for k in reconstructed if k.startswith("mtp.")]
        if mtp_keys:
            print(f"\n  Removing {len(mtp_keys)} MTP keys from reconstructed (not in merged)")
            for k in mtp_keys:
                del reconstructed[k]

    merged_keys = set(merged.keys())
    recon_keys = set(reconstructed.keys())
    common = merged_keys & recon_keys
    only_merged = merged_keys - recon_keys
    only_recon = recon_keys - merged_keys
    print(
        f"\nKey comparison: merged={len(merged_keys)}, reconstructed={len(recon_keys)}, "
        f"common={len(common)}"
    )
    if only_merged:
        print(f"  Keys only in merged ({len(only_merged)}):")
        for k in sorted(only_merged)[:10]:
            print(f"    - {k}")
        if len(only_merged) > 10:
            print(f"    ... and {len(only_merged) - 10} more")
    if only_recon:
        print(f"  Keys only in reconstructed ({len(only_recon)}):")
        for k in sorted(only_recon)[:10]:
            print(f"    - {k}")
        if len(only_recon) > 10:
            print(f"    ... and {len(only_recon) - 10} more")
    if only_merged or only_recon:
        print("  ERROR: key mismatch between merged and reconstructed!")
        return False
    print("  All keys match between merged and reconstructed.")

    if adapted_keys is None:
        adapted_keys = set()

    mismatch_count = 0
    for key in sorted(common):
        m = merged[key]
        r = reconstructed[key]
        if m.shape != r.shape:
            print(f"  SHAPE MISMATCH: {key}")
            mismatch_count += 1
            continue
        if not torch.equal(m, r):
            diff = (m.cuda().float() - r.cuda().float()).abs().max().item()
            mismatch_count += 1
            print(f"  NOT EQUAL: {key}: max_abs_diff={diff:.6e}")

    print(f"\nResults: compared {len(common)} parameters")
    print(f"  Parameters NOT torch.equal: {mismatch_count}")
    return mismatch_count == 0


# ============================================================
#  Qwen2 (dense LLM, no vision, no MoE)
# ============================================================

def compare_qwen2(original_path, merged_path, adapter_path):
    """Verify: original + adapter == merged for Qwen2.5 dense models."""
    print(f"Loading original base weights from: {original_path}")
    print(f"Loading merged export from: {merged_path}")
    print(f"Loading adapter from: {adapter_path}")

    merged = _load_safetensors_dir(merged_path)
    reconstructed = _load_base_as_bf16(original_path)
    lora_pairs, scaling = _parse_lora_pairs(adapter_path)
    hf_config = _load_hf_config(original_path)

    for module_path, weights in lora_pairs.items():
        if "lora_A" not in weights or "lora_B" not in weights:
            continue
        lora_a = weights["lora_A"].cuda()
        lora_b = weights["lora_B"].cuda()
        hf_prefix = module_path.replace("base_model.model.", "", 1)
        delta = scaling * (lora_b @ lora_a)

        # --- Fused QKV adapter (qkv_proj) ---
        if hf_prefix.endswith(".qkv_proj"):
            parent_prefix = hf_prefix.rsplit(".", 1)[0]
            q_key = f"{parent_prefix}.q_proj.weight"
            k_key = f"{parent_prefix}.k_proj.weight"
            v_key = f"{parent_prefix}.v_proj.weight"
            if q_key in reconstructed:
                q_w = reconstructed[q_key].cuda()
                k_w = reconstructed[k_key].cuda()
                v_w = reconstructed[v_key].cuda()
                fused = fuse_qkv(q_w, k_w, v_w, hf_config)
                fused = fused + delta
                q_new, k_new, v_new = split_qkv(fused, hf_config, q_w.shape[0], k_w.shape[0])
                reconstructed[q_key] = q_new.cpu()
                reconstructed[k_key] = k_new.cpu()
                reconstructed[v_key] = v_new.cpu()
                del q_w, k_w, v_w, fused, q_new, k_new, v_new
            continue

        # --- Fused gate_up_proj adapter ---
        if hf_prefix.endswith(".gate_up_proj"):
            parent_prefix = hf_prefix.rsplit(".", 1)[0]
            gate_key = f"{parent_prefix}.gate_proj.weight"
            up_key = f"{parent_prefix}.up_proj.weight"
            child_keys = [k for k in [gate_key, up_key] if k in reconstructed]
            if child_keys:
                total_rows = sum(reconstructed[ck].shape[0] for ck in child_keys)
                if total_rows == delta.shape[0]:
                    parts_cat = torch.cat(
                        [reconstructed[ck].cuda() for ck in child_keys], dim=0
                    )
                    parts_cat = parts_cat + delta
                    offset = 0
                    for ck in child_keys:
                        sz = reconstructed[ck].shape[0]
                        reconstructed[ck] = parts_cat[offset:offset + sz].cpu()
                        offset += sz
                    del parts_cat
            continue

        # --- Standard key lookup ---
        base_key = hf_prefix + ".weight"
        if base_key not in reconstructed:
            print(f"  WARNING: base key {base_key} not found, skipping")
            continue

        reconstructed[base_key] = (reconstructed[base_key].cuda() + delta).cpu()

    return _verify_and_compare(merged, reconstructed)


# ============================================================
#  Qwen3-VL (VLM with ViT QKV interleaving, no MoE)
# ============================================================

def _vit_qkv_hf_to_mcore(qkv, num_heads, head_dim):
    """Convert ViT QKV from HF sequential [Q,K,V] to mcore interleaved per-head."""
    hidden_dim = qkv.shape[-1]
    return (
        qkv.view(3, num_heads, -1, head_dim, hidden_dim)
        .transpose(0, 1)
        .flatten(1, 2)
        .reshape(-1, hidden_dim)
        .contiguous()
    )


def _vit_qkv_mcore_to_hf(qkv, num_heads, head_dim):
    """Convert ViT QKV from mcore interleaved per-head to HF sequential [Q,K,V]."""
    hidden_dim = qkv.shape[-1]
    return (
        qkv.view(num_heads, 3, -1, head_dim, hidden_dim)
        .transpose(0, 1)
        .reshape(-1, hidden_dim)
        .contiguous()
    )


def compare_qwen3_vl(original_path, merged_path, adapter_path):
    """Verify: original + adapter == merged for Qwen3-VL models.

    Handles:
    - ViT QKV interleaving (mcore uses per-head interleaved, HF uses [Q,K,V])
    - Fused language model adapters (qkv_proj, gate_up_proj)
    """
    print(f"Loading original base weights from: {original_path}")
    print(f"Loading merged export from: {merged_path}")
    print(f"Loading adapter from: {adapter_path}")

    merged = _load_safetensors_dir(merged_path)
    reconstructed = _load_base_as_bf16(original_path)
    lora_pairs, scaling = _parse_lora_pairs(adapter_path)
    hf_config = _load_hf_config(original_path)
    vision_config = _load_vision_config(original_path)

    # --- Pre-process: group canonical ViT QKV adapters ---
    # Canonical LoRA exports separate q_proj/k_proj/v_proj adapters for ViT
    # qkv layers, but the HF base model stores a single fused attn.qkv.weight.
    # Group them by parent module and apply the stacked delta together.
    _vit_qkv_groups: dict = {}
    for module_path in list(lora_pairs.keys()):
        hf_prefix = module_path.replace("base_model.model.", "", 1)
        if (
            vision_config
            and "visual" in hf_prefix
            and ".attn.qkv." in hf_prefix
        ):
            parent = hf_prefix.rsplit(".", 1)[0]  # e.g. model.visual.blocks.0.attn.qkv
            fused_base_key = parent + ".weight"
            if fused_base_key in reconstructed:
                _vit_qkv_groups.setdefault(parent, []).append(module_path)

    for parent, module_paths in _vit_qkv_groups.items():
        q_delta = k_delta = v_delta = None
        for mp in module_paths:
            weights = lora_pairs.pop(mp)
            if "lora_A" not in weights or "lora_B" not in weights:
                continue
            lora_a = weights["lora_A"].cuda()
            lora_b = weights["lora_B"].cuda()
            delta = scaling * (lora_b @ lora_a)
            hf_prefix = mp.replace("base_model.model.", "", 1)
            component = hf_prefix.rsplit(".", 1)[1]  # "q_proj", "k_proj", or "v_proj"
            if component == "q_proj":
                q_delta = delta
            elif component == "k_proj":
                k_delta = delta
            elif component == "v_proj":
                v_delta = delta
        deltas = [d for d in [q_delta, k_delta, v_delta] if d is not None]
        if deltas:
            fused_delta = torch.cat(deltas, dim=0)
            fused_key = parent + ".weight"
            reconstructed[fused_key] = (reconstructed[fused_key].cuda() + fused_delta).cpu()
            del fused_delta
        torch.cuda.empty_cache()

    for module_path, weights in lora_pairs.items():
        if "lora_A" not in weights or "lora_B" not in weights:
            continue
        lora_a = weights["lora_A"].cuda()
        lora_b = weights["lora_B"].cuda()
        hf_prefix = module_path.replace("base_model.model.", "", 1)
        delta = scaling * (lora_b @ lora_a)

        # --- Fused QKV adapter (language model) ---
        if hf_prefix.endswith(".qkv_proj"):
            parent_prefix = hf_prefix.rsplit(".", 1)[0]
            q_key = f"{parent_prefix}.q_proj.weight"
            k_key = f"{parent_prefix}.k_proj.weight"
            v_key = f"{parent_prefix}.v_proj.weight"
            if q_key in reconstructed:
                q_w = reconstructed[q_key].cuda()
                k_w = reconstructed[k_key].cuda()
                v_w = reconstructed[v_key].cuda()
                fused = fuse_qkv(q_w, k_w, v_w, hf_config)
                fused = fused + delta
                q_new, k_new, v_new = split_qkv(
                    fused, hf_config, q_w.shape[0], k_w.shape[0]
                )
                reconstructed[q_key] = q_new.cpu()
                reconstructed[k_key] = k_new.cpu()
                reconstructed[v_key] = v_new.cpu()
                del q_w, k_w, v_w, fused, q_new, k_new, v_new
            continue

        # --- Fused gate_up_proj adapter (language model) ---
        if hf_prefix.endswith(".gate_up_proj"):
            parent_prefix = hf_prefix.rsplit(".", 1)[0]
            gate_key = f"{parent_prefix}.gate_proj.weight"
            up_key = f"{parent_prefix}.up_proj.weight"
            child_keys = [k for k in [gate_key, up_key] if k in reconstructed]
            if child_keys:
                total_rows = sum(reconstructed[ck].shape[0] for ck in child_keys)
                if total_rows == delta.shape[0]:
                    parts_cat = torch.cat(
                        [reconstructed[ck].cuda() for ck in child_keys], dim=0
                    )
                    parts_cat = parts_cat + delta
                    offset = 0
                    for ck in child_keys:
                        sz = reconstructed[ck].shape[0]
                        reconstructed[ck] = parts_cat[offset:offset + sz].cpu()
                        offset += sz
                    del parts_cat
            continue

        # --- Standard key lookup ---
        base_key = hf_prefix + ".weight"
        if base_key not in reconstructed:
            print(f"  WARNING: base key {base_key} not found, skipping")
            continue

        # ViT QKV: adapter delta is in mcore interleaved space,
        # base weight is in HF sequential [Q,K,V] format.
        if (
            vision_config
            and ".attn.qkv.weight" in base_key
            and "visual" in base_key
        ):
            vit_num_heads = vision_config["num_heads"]
            vit_hidden = vision_config.get(
                "embed_dim", vision_config.get("hidden_size")
            )
            vit_head_dim = vit_hidden // vit_num_heads
            base_w = reconstructed[base_key].cuda()
            base_mcore = _vit_qkv_hf_to_mcore(base_w, vit_num_heads, vit_head_dim)
            merged_mcore = base_mcore + delta
            reconstructed[base_key] = _vit_qkv_mcore_to_hf(
                merged_mcore, vit_num_heads, vit_head_dim
            ).cpu()
            del base_w, base_mcore, merged_mcore
            continue

        reconstructed[base_key] = (reconstructed[base_key].cuda() + delta).cpu()

    return _verify_and_compare(merged, reconstructed)


# ============================================================
#  Qwen3.5 MoE - lora type
# ============================================================

def _has_output_gate(hf_config):
    return hf_config.get("attn_output_gate", False) or hf_config.get(
        "attention_output_gate", False
    )


def _parse_fused_order(fused_suffix, available_stems):
    """Parse a fused module name into ordered component stems."""
    remaining = fused_suffix
    result = []
    stems_by_len = sorted(available_stems, key=len, reverse=True)
    while remaining:
        matched = False
        for stem in stems_by_len:
            if stem in result:
                continue
            if remaining == stem:
                result.append(stem)
                remaining = ""
                matched = True
                break
            elif remaining.startswith(stem + "_"):
                result.append(stem)
                remaining = remaining[len(stem) + 1:]
                matched = True
                break
        if not matched:
            return None
    return result


def compare_qwen3_5(original_path, merged_path, adapter_path):
    """Verify: original + adapter == merged for Qwen3.5 MoE with type=lora.

    Handles:
    - output_gate: q_proj is [Q,G] interleaved, delta only applies to Q half
    - 3D expert weights: adapter is 2D but base may be 3D stacked [E, H, I]
    - MTP removal: mtp layers not built during training
    - Fused adapter names (e.g. in_proj_qkv_in_proj_z_in_proj_b_in_proj_a)
    """
    print(f"Loading original base weights from: {original_path}")
    print(f"Loading merged export from: {merged_path}")
    print(f"Loading adapter from: {adapter_path}")

    merged = _load_safetensors_dir(merged_path)
    reconstructed = _load_base_as_bf16(original_path)
    lora_pairs, scaling = _parse_lora_pairs(adapter_path)
    hf_config = _load_hf_config(original_path)

    has_gate = _has_output_gate(hf_config)
    num_heads = hf_config["num_attention_heads"]
    head_dim = _get_head_dim(hf_config)

    # Collect per-expert deltas for batch application
    expert_deltas = {}

    for module_path, weights in lora_pairs.items():
        if "lora_A" not in weights or "lora_B" not in weights:
            continue
        lora_a = weights["lora_A"].cuda()
        lora_b = weights["lora_B"].cuda()
        hf_prefix = module_path.replace("base_model.model.", "", 1)
        base_key = hf_prefix + ".weight"
        delta = scaling * (lora_b @ lora_a)

        # --- Fused QKV adapter (qkv_proj) ---
        if hf_prefix.endswith(".qkv_proj"):
            parent_prefix = hf_prefix.rsplit(".", 1)[0]
            q_key = f"{parent_prefix}.q_proj.weight"
            k_key = f"{parent_prefix}.k_proj.weight"
            v_key = f"{parent_prefix}.v_proj.weight"
            if q_key in reconstructed:
                q_w = reconstructed[q_key].cuda()
                k_w = reconstructed[k_key].cuda()
                v_w = reconstructed[v_key].cuda()
                fused = fuse_qkv(q_w, k_w, v_w, hf_config)
                fused = fused + delta
                q_new, k_new, v_new = split_qkv(fused, hf_config, q_w.shape[0], k_w.shape[0])
                reconstructed[q_key] = q_new.cpu()
                reconstructed[k_key] = k_new.cpu()
                reconstructed[v_key] = v_new.cpu()
                del q_w, k_w, v_w, fused, q_new, k_new, v_new
            continue

        # --- Fused gate_up_proj adapter ---
        if hf_prefix.endswith(".gate_up_proj"):
            parent_prefix = hf_prefix.rsplit(".", 1)[0]
            gate_key = f"{parent_prefix}.gate_proj.weight"
            up_key = f"{parent_prefix}.up_proj.weight"
            child_keys = [k for k in [gate_key, up_key] if k in reconstructed]
            if child_keys:
                total_rows = sum(reconstructed[ck].shape[0] for ck in child_keys)
                if total_rows == delta.shape[0]:
                    parts_cat = torch.cat(
                        [reconstructed[ck].cuda() for ck in child_keys], dim=0
                    )
                    parts_cat = parts_cat + delta
                    offset = 0
                    for ck in child_keys:
                        sz = reconstructed[ck].shape[0]
                        reconstructed[ck] = parts_cat[offset:offset + sz].cpu()
                        offset += sz
                    del parts_cat
            continue

        # --- Per-expert adapter: collect delta, defer GPU work ---
        parts = hf_prefix.rsplit(".", 1)
        if len(parts) == 2 and parts[1].isdigit():
            expert_id = int(parts[1])
            base_3d_key = parts[0]
            if base_3d_key not in reconstructed:
                base_3d_key_w = base_3d_key + ".weight"
                if base_3d_key_w in reconstructed:
                    base_3d_key = base_3d_key_w
            if base_3d_key in reconstructed and reconstructed[base_3d_key].ndim == 3:
                expert_deltas.setdefault(base_3d_key, {})[expert_id] = delta.cpu()
                continue

        # --- Key exists directly ---
        if base_key in reconstructed:
            # output_gate: q_proj base is [Q,G] interleaved, delta only affects Q
            if (
                has_gate
                and "q_proj" in base_key
                and reconstructed[base_key].shape[0] == 2 * delta.shape[0]
            ):
                base_w = reconstructed[base_key].cuda().reshape(num_heads, 2 * head_dim, -1)
                q_part = base_w[:, :head_dim, :].reshape(-1, base_w.shape[-1])
                q_part = q_part + delta
                base_w[:, :head_dim, :] = q_part.reshape(num_heads, head_dim, -1)
                reconstructed[base_key] = base_w.reshape(-1, base_w.shape[-1]).cpu()
                del base_w, q_part
                continue

            if delta.shape[0] != reconstructed[base_key].shape[0]:
                print(f"  WARNING: shape mismatch for {base_key}, skipping")
                continue

            reconstructed[base_key] = (reconstructed[base_key].cuda() + delta).cpu()
            continue

        # --- Key not found: try expert 3D broadcast (legacy shared adapter) ---
        if delta.ndim == 2:
            expert_handled = False
            for suffix in ("down_proj", "gate_up_proj"):
                candidate = f"{hf_prefix}.{suffix}"
                if candidate in reconstructed and reconstructed[candidate].ndim == 3:
                    base_3d = reconstructed[candidate]
                    if base_3d.shape[1:] == delta.shape:
                        reconstructed[candidate] = (
                            base_3d.cuda() + delta.unsqueeze(0)
                        ).cpu()
                        expert_handled = True
                        break
            if expert_handled:
                continue

        # --- Key not found: try generic fused multi-weight ---
        parent_prefix = hf_prefix.rsplit(".", 1)[0] if "." in hf_prefix else ""
        fused_module = hf_prefix.rsplit(".", 1)[1] if "." in hf_prefix else ""
        if parent_prefix and fused_module:
            candidate_keys = [
                k for k in reconstructed
                if k.startswith(parent_prefix + ".") and k.endswith(".weight")
                and k.count(".") == parent_prefix.count(".") + 2
                and reconstructed[k].ndim == 2
            ]
            stems = [
                k[len(parent_prefix) + 1:].replace(".weight", "")
                for k in candidate_keys
            ]
            ordered_stems = _parse_fused_order(fused_module, stems)
            if ordered_stems:
                child_keys = [f"{parent_prefix}.{s}.weight" for s in ordered_stems]
                total_rows = sum(reconstructed[ck].shape[0] for ck in child_keys)
                if total_rows == delta.shape[0]:
                    parts_cat = torch.cat(
                        [reconstructed[ck].cuda() for ck in child_keys], dim=0
                    )
                    parts_cat = parts_cat + delta
                    offset = 0
                    for ck in child_keys:
                        sz = reconstructed[ck].shape[0]
                        reconstructed[ck] = parts_cat[offset:offset + sz].cpu()
                        offset += sz
                    del parts_cat
                    continue

        print(f"  WARNING: base key {base_key} not found, skipping")

    # --- Batch-apply collected per-expert deltas ---
    if expert_deltas:
        print(f"  Applying per-expert deltas for {len(expert_deltas)} 3D tensors...")
        for base_3d_key, deltas_by_id in expert_deltas.items():
            base_3d = reconstructed[base_3d_key].cuda()
            for expert_id, delta_cpu in deltas_by_id.items():
                base_3d[expert_id] += delta_cpu.cuda()
            reconstructed[base_3d_key] = base_3d.cpu()
            del base_3d
        torch.cuda.empty_cache()

    return _verify_and_compare(merged, reconstructed)


# ============================================================
#  Qwen3.5 MoE - canonical_lora type
# ============================================================

def fuse_qkv(q, k, v, hf_config):
    """Fuse separate Q, K, V into Megatron-Core interleaved QKV layout."""
    hidden_dim = hf_config["hidden_size"]
    head_dim = _get_head_dim(hf_config)
    num_attention_heads = hf_config["num_attention_heads"]
    num_kv_heads = hf_config["num_key_value_heads"]
    group_dim = head_dim * num_attention_heads // num_kv_heads

    if _has_output_gate(hf_config):
        combined_w = q.reshape(num_attention_heads, 2 * head_dim, -1)
        q_w = combined_w.narrow(1, 0, head_dim).reshape(num_attention_heads * head_dim, -1)
        g_w = combined_w.narrow(1, head_dim, head_dim).reshape(num_attention_heads * head_dim, -1)
        q_w = q_w.view(num_kv_heads, group_dim, -1)
        g_w = g_w.view(num_kv_heads, group_dim, -1)
        k = k.view(num_kv_heads, head_dim, -1)
        v = v.view(num_kv_heads, head_dim, -1)
        return torch.cat([q_w, g_w, k, v], dim=1).view(-1, hidden_dim).contiguous()
    else:
        real_num_kv_heads = q.shape[0] // group_dim
        q = q.view(real_num_kv_heads, group_dim, -1)
        k = k.view(real_num_kv_heads, head_dim, -1)
        v = v.view(real_num_kv_heads, head_dim, -1)
        return torch.cat([q, k, v], dim=1).view(-1, hidden_dim).contiguous()


def split_qkv(qkv, hf_config, q_rows, k_rows):
    """Split Megatron-Core interleaved QKV into separate Q, K, V."""
    hidden_dim = qkv.shape[-1]
    head_dim = _get_head_dim(hf_config)
    num_attention_heads = hf_config["num_attention_heads"]
    num_kv_heads = hf_config["num_key_value_heads"]
    group_dim = head_dim * num_attention_heads // num_kv_heads

    if _has_output_gate(hf_config):
        qkv = qkv.view(num_kv_heads, -1, hidden_dim)
        q_w = qkv[:, :group_dim].reshape(-1, hidden_dim)
        g_w = qkv[:, group_dim:2 * group_dim].reshape(-1, hidden_dim)
        k = qkv[:, 2 * group_dim:2 * group_dim + head_dim].reshape(-1, hidden_dim)
        v = qkv[:, 2 * group_dim + head_dim:].reshape(-1, hidden_dim)
        q_w = q_w.reshape(num_attention_heads, head_dim, -1)
        g_w = g_w.reshape(num_attention_heads, head_dim, -1)
        q_proj = torch.cat([q_w, g_w], dim=1).reshape(num_attention_heads * 2 * head_dim, -1)
        return q_proj, k, v
    else:
        qkv = qkv.view(num_kv_heads, -1, hidden_dim)
        q = qkv[:, :group_dim].reshape(-1, hidden_dim)
        k = qkv[:, group_dim:group_dim + head_dim].reshape(-1, hidden_dim)
        v = qkv[:, group_dim + head_dim:].reshape(-1, hidden_dim)
        return q, k, v


def compare_qwen3_5_moe(original_path, merged_path, adapter_path):
    """Verify: original + adapter == merged for Qwen3.5 MoE with type=canonical_lora.

    Handles:
    - output_gate: q_proj is [Q,G] interleaved, delta only applies to Q half
    - Fused QKV adapters (qkv_proj → fuse/split Q,K,V)
    - Fused gate_up_proj adapters
    - 3D expert weights
    - MTP removal
    """
    print(f"Loading original base weights from: {original_path}")
    print(f"Loading merged export from: {merged_path}")
    print(f"Loading adapter from: {adapter_path}")

    merged = _load_safetensors_dir(merged_path)
    reconstructed = _load_base_as_bf16(original_path)
    lora_pairs, scaling = _parse_lora_pairs(adapter_path)
    hf_config = _load_hf_config(original_path)

    has_gate = _has_output_gate(hf_config)
    num_heads = hf_config["num_attention_heads"]
    head_dim = _get_head_dim(hf_config)
    adapted_keys = set()

    # Collect per-expert deltas first, then apply in batch (one GPU round-trip
    # per 3D tensor instead of one per expert).
    # expert_deltas: { base_3d_key: { expert_id: delta_cpu } }
    expert_deltas = {}

    for module_path, weights in lora_pairs.items():
        if "lora_A" not in weights or "lora_B" not in weights:
            continue
        lora_a = weights["lora_A"].cuda()
        lora_b = weights["lora_B"].cuda()
        hf_prefix = module_path.replace("base_model.model.", "", 1)
        delta = scaling * (lora_b @ lora_a)

        # --- Fused QKV adapter (qkv_proj) ---
        if hf_prefix.endswith(".qkv_proj"):
            parent_prefix = hf_prefix.rsplit(".", 1)[0]
            q_key = f"{parent_prefix}.q_proj.weight"
            k_key = f"{parent_prefix}.k_proj.weight"
            v_key = f"{parent_prefix}.v_proj.weight"
            if q_key in reconstructed:
                q_w = reconstructed[q_key].cuda()
                k_w = reconstructed[k_key].cuda()
                v_w = reconstructed[v_key].cuda()
                fused = fuse_qkv(q_w, k_w, v_w, hf_config)
                fused = fused + delta
                q_new, k_new, v_new = split_qkv(fused, hf_config, q_w.shape[0], k_w.shape[0])
                reconstructed[q_key] = q_new.cpu()
                reconstructed[k_key] = k_new.cpu()
                reconstructed[v_key] = v_new.cpu()
                adapted_keys.update([q_key, k_key, v_key])
                del q_w, k_w, v_w, fused, q_new, k_new, v_new
            continue

        # --- Fused gate_up_proj adapter ---
        if hf_prefix.endswith(".gate_up_proj"):
            parent_prefix = hf_prefix.rsplit(".", 1)[0]
            gate_key = f"{parent_prefix}.gate_proj.weight"
            up_key = f"{parent_prefix}.up_proj.weight"
            child_keys = [k for k in [gate_key, up_key] if k in reconstructed]
            if child_keys:
                total_rows = sum(reconstructed[ck].shape[0] for ck in child_keys)
                if total_rows == delta.shape[0]:
                    parts_cat = torch.cat(
                        [reconstructed[ck].cuda() for ck in child_keys], dim=0
                    )
                    parts_cat = parts_cat + delta
                    offset = 0
                    for ck in child_keys:
                        sz = reconstructed[ck].shape[0]
                        reconstructed[ck] = parts_cat[offset:offset + sz].cpu()
                        offset += sz
                    adapted_keys.update(child_keys)
                    del parts_cat
            continue

        # --- Per-expert adapter: collect delta, defer GPU work ---
        parts = hf_prefix.rsplit(".", 1)
        if len(parts) == 2 and parts[1].isdigit():
            expert_id = int(parts[1])
            base_3d_key = parts[0]
            if base_3d_key not in reconstructed:
                base_3d_key_w = base_3d_key + ".weight"
                if base_3d_key_w in reconstructed:
                    base_3d_key = base_3d_key_w
            if base_3d_key in reconstructed and reconstructed[base_3d_key].ndim == 3:
                expert_deltas.setdefault(base_3d_key, {})[expert_id] = delta.cpu()
                adapted_keys.add(base_3d_key)
                continue

        # --- Standard separate adapter ---
        base_key = hf_prefix + ".weight"

        if base_key in reconstructed:
            # output_gate: q_proj base is [Q,G] interleaved, delta only affects Q
            if (
                has_gate
                and "q_proj" in base_key
                and reconstructed[base_key].shape[0] == 2 * delta.shape[0]
            ):
                base_w = reconstructed[base_key].cuda().reshape(num_heads, 2 * head_dim, -1)
                q_part = base_w[:, :head_dim, :].reshape(-1, base_w.shape[-1])
                q_part = q_part + delta
                base_w[:, :head_dim, :] = q_part.reshape(num_heads, head_dim, -1)
                reconstructed[base_key] = base_w.reshape(-1, base_w.shape[-1]).cpu()
                adapted_keys.add(base_key)
                del base_w, q_part
                continue

            if delta.shape[0] != reconstructed[base_key].shape[0]:
                print(f"  WARNING: shape mismatch for {base_key}, skipping")
                continue

            reconstructed[base_key] = (reconstructed[base_key].cuda() + delta).cpu()
            adapted_keys.add(base_key)
            continue

        # --- Expert 3D broadcast (legacy: shared adapter for all experts) ---
        if delta.ndim == 2:
            expert_handled = False
            for suffix in ("down_proj", "gate_up_proj"):
                candidate = f"{hf_prefix}.{suffix}"
                if candidate in reconstructed and reconstructed[candidate].ndim == 3:
                    base_3d = reconstructed[candidate]
                    if base_3d.shape[1:] == delta.shape:
                        reconstructed[candidate] = (
                            base_3d.cuda() + delta.unsqueeze(0)
                        ).cpu()
                        adapted_keys.add(candidate)
                        expert_handled = True
                        break
            if expert_handled:
                continue

        # --- Generic fused multi-weight (e.g. in_proj_qkv_in_proj_z_...) ---
        parent_prefix = hf_prefix.rsplit(".", 1)[0] if "." in hf_prefix else ""
        fused_module = hf_prefix.rsplit(".", 1)[1] if "." in hf_prefix else ""
        if parent_prefix and fused_module:
            candidate_keys = [
                k for k in reconstructed
                if k.startswith(parent_prefix + ".") and k.endswith(".weight")
                and k.count(".") == parent_prefix.count(".") + 2
                and reconstructed[k].ndim == 2
            ]
            stems = [
                k[len(parent_prefix) + 1:].replace(".weight", "")
                for k in candidate_keys
            ]
            ordered_stems = _parse_fused_order(fused_module, stems)
            if ordered_stems:
                child_keys = [f"{parent_prefix}.{s}.weight" for s in ordered_stems]
                total_rows = sum(reconstructed[ck].shape[0] for ck in child_keys)
                if total_rows == delta.shape[0]:
                    parts_cat = torch.cat(
                        [reconstructed[ck].cuda() for ck in child_keys], dim=0
                    )
                    parts_cat = parts_cat + delta
                    offset = 0
                    for ck in child_keys:
                        sz = reconstructed[ck].shape[0]
                        reconstructed[ck] = parts_cat[offset:offset + sz].cpu()
                        offset += sz
                    adapted_keys.update(child_keys)
                    del parts_cat
                    continue

        print(f"  WARNING: base key {base_key} not found, skipping")

    # --- Batch-apply collected per-expert deltas (one GPU round-trip per 3D key) ---
    if expert_deltas:
        print(f"  Applying per-expert deltas for {len(expert_deltas)} 3D tensors...")
        for base_3d_key, deltas_by_id in expert_deltas.items():
            base_3d = reconstructed[base_3d_key].cuda()
            for expert_id, delta_cpu in deltas_by_id.items():
                base_3d[expert_id] += delta_cpu.cuda()
            reconstructed[base_3d_key] = base_3d.cpu()
            del base_3d
        torch.cuda.empty_cache()

    return _verify_and_compare(merged, reconstructed, adapted_keys)


# ============================================================
#  Generic entry (backwards compat)
# ============================================================

def compare_adapter_merge(original_path, merged_path, adapter_path):
    """Auto-detect model type and dispatch to the correct compare function."""
    hf_config_path = os.path.join(original_path, "config.json")
    with open(hf_config_path) as f:
        cfg = json.load(f)

    # Resolve model_type: check top-level first, then text_config
    model_type = cfg.get("model_type", "")
    if not model_type or model_type in ("qwen3_5_vl",):
        model_type = cfg.get("text_config", {}).get("model_type", model_type)

    if model_type in ("qwen3_vl", "qwen2_vl"):
        return compare_qwen3_vl(original_path, merged_path, adapter_path)
    elif model_type in ("qwen3_5_moe", "qwen3_5"):
        adapter_state = load_file(
            os.path.join(adapter_path, "adapter_model.safetensors"), device="cpu"
        )
        # canonical_lora exports separate q_proj/k_proj/v_proj adapters;
        # standard lora exports a single fused qkv_proj adapter.
        has_separate_qkv = any(".q_proj.lora_" in k for k in adapter_state)
        if has_separate_qkv:
            return compare_qwen3_5_moe(original_path, merged_path, adapter_path)
        else:
            return compare_qwen3_5(original_path, merged_path, adapter_path)
    else:
        return compare_qwen2(original_path, merged_path, adapter_path)


# ============================================================
#  unittest-based integration tests
# ============================================================

class TestLoRAMbridge(unittest.IsolatedAsyncioTestCase):

    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    async def _run_and_verify(self, lora_type: str, tp_size: int):
        """Train 1 step, verify metrics directly, then verify checkpoint."""
        label = f"{lora_type}/tp{tp_size}"
        ckpt_path = f"test_lora_mbridge_{lora_type}_tp{tp_size}"

        if os.path.exists(ckpt_path):
            shutil.rmtree(ckpt_path)

        config = load_config("test_lora_mbridge", FinetuneConfig)
        config.policy.lora.type = lora_type
        config.training.exit_step = 1
        config.training.save_interval = 1
        config.policy.dist_config.tensor_model_parallel_size = tp_size
        config.checkpoint.save_ckpt_path = ckpt_path
        config.checkpoint.load_ckpt_path = ckpt_path

        try:
            trainer = FinetuneTrainer()
            all_actor_metrics = await trainer.launch_then_run_with_recovery(config)

            # 1) Verify loss / grad_norm from returned metrics
            assert all_actor_metrics is not None, f"[{label}] No metrics returned"
            has_metrics = False
            for actor_metrics in all_actor_metrics:
                if not actor_metrics:
                    continue
                has_metrics = True
                for step_metric in actor_metrics:
                    assert "finetune/lm_loss" in step_metric, (
                        f"[{label}] finetune/lm_loss missing from metrics"
                    )
                    assert "finetune/grad_norm" in step_metric, (
                        f"[{label}] finetune/grad_norm missing from metrics"
                    )

                    lm_loss = step_metric["finetune/lm_loss"]
                    grad_norm = step_metric["finetune/grad_norm"]

                    assert lm_loss == lm_loss, f"[{label}] NaN lm_loss"
                    assert 0.01 <= lm_loss <= 1.0, (
                        f"[{label}] lm_loss={lm_loss:.4f} out of range "
                    )

                    assert grad_norm == grad_norm, f"[{label}] NaN grad_norm"
                    assert 0 <= grad_norm <= 2.0, (
                        f"[{label}] grad_norm={grad_norm:.4f} out of range "
                    )

                    print(
                        f"[{label}] loss={lm_loss:.4f}  "
                        f"grad_norm={grad_norm:.4f}"
                    )

            assert has_metrics, f"[{label}] No actor returned training metrics"

            # 2) Verify checkpoint existence
            hf_dir = os.path.join(ckpt_path, "hf")
            assert os.path.isdir(hf_dir), f"[{label}] HF export dir missing: {hf_dir}"

            ckpt_steps = sorted(
                [
                    d for d in os.listdir(hf_dir)
                    if os.path.isdir(os.path.join(hf_dir, d)) and d.isdigit()
                ],
                key=int,
            )
            assert len(ckpt_steps) > 0, f"[{label}] No merged HF checkpoint dirs found"
            latest_step = ckpt_steps[-1]

            base_path = os.path.join(hf_dir, latest_step)
            merged_path = os.path.join(hf_dir, f"{latest_step}_merge")
            adapter_path = os.path.join(hf_dir, f"{latest_step}_adapter")
            assert os.path.isdir(base_path), f"[{label}] Base dir missing: {base_path}"
            assert os.path.isdir(merged_path), f"[{label}] Merged dir missing: {merged_path}"
            assert os.path.isdir(adapter_path), f"[{label}] Adapter dir missing: {adapter_path}"

            # 3) Verify: original + adapter == merged
            print(f"[{label}] Verifying adapter merge correctness (step {latest_step})...")
            hf_original_path = config.policy.hf_model_path
            merge_ok = compare_adapter_merge(hf_original_path, merged_path, adapter_path)
            assert merge_ok, (
                f"[{label}] Adapter merge verification failed: "
                f"original + adapter != merged (large diffs found) at step {latest_step}"
            )
            merge_ok = compare_adapter_merge(base_path, merged_path, adapter_path)
            assert merge_ok, (
                f"[{label}] Adapter merge verification failed: "
                f"original + adapter != merged (large diffs found) at step {latest_step}"
            )
            print(f"[{label}] Adapter merge verification PASSED")

        finally:
            if os.path.exists(ckpt_path):
                shutil.rmtree(ckpt_path)

    async def test_lora_tp1(self):
        await self._run_and_verify("lora", tp_size=1)

    async def test_lora_tp2(self):
        await self._run_and_verify("lora", tp_size=2)

    async def test_canonical_lora_tp1(self):
        await self._run_and_verify("canonical_lora", tp_size=1)

    async def test_canonical_lora_tp2(self):
        await self._run_and_verify("canonical_lora", tp_size=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", default="hf-hub/Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--merged", default="ckpt_lora_tp_test_tp2/hf/3357")
    parser.add_argument("--adapter", default="ckpt_lora_tp_test_tp2/hf/3357_adapter")
    args = parser.parse_args()
    res = compare_adapter_merge(args.original, args.merged, args.adapter)
    print(f"Result: {res}")
