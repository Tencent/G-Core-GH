import math
import random

import torch
from PIL import Image
from torch.nn.attention.flex_attention import and_masks, or_masks


def create_sparse_mask(
    document_lens,
    split_lens,
    attn_modes,
    device,
    vae_token_indexes=None,
    total_length=None,
    enable_vae_mask=True
):
    """
    创建稀疏注意力mask

    Args:
        document_lens: 每个document的长度
        split_lens: 每个split的长度
        attn_modes: 每个split的attention模式 ('causal', 'full', 'noise')
        device: torch device
        vae_token_indexes: VAE tokens的索引位置（可选，用于VAE isolation）
        total_length: 序列总长度（当提供vae_token_indexes时需要）
        enable_vae_mask: 是否启用VAE isolation mask（默认True，兼容旧行为）
    """
    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    def full_and_noise_mask(b, h, q_idx, kv_idx):
        return (full_and_noise_seq_id[q_idx]
                == full_and_noise_seq_id[kv_idx]) & (full_and_noise_seq_id[q_idx] >= 0)

    def remove_noise_mask(b, h, q_idx, kv_idx):
        return (~((noise_seq_id[kv_idx] >= 0) & (noise_seq_id[q_idx] != noise_seq_id[kv_idx])))

    def sample_mask(b, h, q_idx, kv_idx):
        return document_id[q_idx] == document_id[kv_idx]

    def vae_mask(b, h, q_idx, kv_idx):
        # VAE的特殊规则：
        # 需要区分：纯VAE（条件输入）vs Noise（生成目标）
        # - 纯VAE: vae_position_mask=True 且 noise_seq_id<0
        # - Noise: vae_position_mask=True 且 noise_seq_id>=0

        q_in_vae_tokens = vae_position_mask[q_idx]
        kv_in_vae_tokens = vae_position_mask[kv_idx]
        q_is_noise = noise_seq_id[q_idx] >= 0
        kv_is_noise = noise_seq_id[kv_idx] >= 0

        # 区分纯VAE和Noise
        q_is_pure_vae = q_in_vae_tokens & (~q_is_noise)
        kv_is_pure_vae = kv_in_vae_tokens & (~kv_is_noise)

        # 使用张量操作替代 if 语句，避免动态控制流
        # 规则1: 如果 kv 是纯VAE，只能被自己（同一个VAE split）或 Noise 看到
        is_same_vae_split = (full_and_noise_seq_id[q_idx] == full_and_noise_seq_id[kv_idx])
        kv_pure_vae_mask = (is_same_vae_split & q_is_pure_vae) | q_is_noise

        # 规则2: 如果 q 是纯VAE，只能看到同一个 VAE split 中的纯VAE
        q_pure_vae_mask = is_same_vae_split & kv_is_pure_vae

        # 其他情况（包括 Noise 和 非VAE tokens）不受 VAE mask 限制
        # 使用 torch.where 或布尔运算组合所有情况
        # 当 kv_is_pure_vae 时，使用 kv_pure_vae_mask
        # 当 q_is_pure_vae 且 kv 不是纯VAE 时，使用 q_pure_vae_mask (False)
        # 其他情况返回 True
        result = torch.where(
            kv_is_pure_vae, kv_pure_vae_mask, torch.where(q_is_pure_vae, q_pure_vae_mask, True)
        )
        return result

    full_and_noise_tmp = []
    noise_tmp = []

    for i, (length, model) in enumerate(zip(split_lens, attn_modes)):
        value = i if model in ['full', 'noise'] else -1
        full_and_noise_tmp.extend([value] * length)
        value_noise = i if model == 'noise' else -1
        noise_tmp.extend([value_noise] * length)

    full_and_noise_seq_id = torch.Tensor(full_and_noise_tmp).to(device)
    noise_seq_id = torch.Tensor(noise_tmp).to(device)

    document_id = torch.cat([torch.full((l, ), i)
                             for i, l in enumerate(document_lens, start=1)]).to(device)

    # 如果启用VAE mask且提供了 VAE token 索引，创建 VAE position mask
    if enable_vae_mask and vae_token_indexes is not None and total_length is not None:
        vae_position_mask = torch.zeros(total_length, dtype=torch.bool, device=device)
        vae_position_mask[vae_token_indexes] = True
        return and_masks(
            or_masks(causal_mask, full_and_noise_mask), remove_noise_mask, sample_mask, vae_mask
        )
    else:

        # 不使用 VAE mask
        return and_masks(or_masks(causal_mask, full_and_noise_mask), remove_noise_mask, sample_mask)


def patchify(image, patch_size):
    p = patch_size
    c, h, w = image.shape
    assert h % p == 0 and w % p == 0
    image = image.reshape(c, h // p, p, w // p, p)
    image = torch.einsum("chpwq->hwpqc", image)
    image = image.reshape(-1, p**2 * c)
    return image


def get_flattened_position_ids_extrapolate(img_h, img_w, patch_size, max_num_patches_per_side):
    num_patches_h, num_patches_w = img_h // patch_size, img_w // patch_size
    coords_h = torch.arange(0, num_patches_h)
    coords_w = torch.arange(0, num_patches_w)
    pos_ids = (coords_h[:, None] * max_num_patches_per_side + coords_w).flatten()
    return pos_ids


def get_flattened_position_ids_interpolate(img_h, img_w, patch_size, max_num_patches_per_side):
    num_patches_h, num_patches_w = img_h // patch_size, img_w // patch_size
    boundaries = torch.arange(1 / max_num_patches_per_side, 1.0, 1 / max_num_patches_per_side)
    fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / num_patches_h)
    fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / num_patches_w)
    bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
    bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)
    pos_ids = (bucket_coords_h[:, None] * max_num_patches_per_side + bucket_coords_w).flatten()
    return pos_ids


def prepare_attention_mask_per_sample(split_lens, attn_modes, device="cpu"):
    """
    Build a dense attention mask for a single packed sample.
    Used by PackedDataset when use_flex=False.
    """
    sample_len = sum(split_lens)
    attention_mask = torch.zeros((sample_len, sample_len), dtype=torch.bool, device=device)

    csum = 0
    for s, attn_mode in zip(split_lens, attn_modes):
        assert attn_mode in ['causal', 'full', 'noise']
        if attn_mode == "causal":
            attention_mask[csum:csum + s, csum:csum + s] = torch.ones((s, s), device=device).tril()
            attention_mask[csum:csum + s, :csum] = 1
        else:
            attention_mask[csum:csum + s, csum:csum + s] = torch.ones((s, s))
            attention_mask[csum:csum + s, :csum] = 1
        csum += s

    csum = 0
    for s, attn_mode in zip(split_lens, attn_modes):
        if attn_mode == "noise":
            attention_mask[:, csum:csum + s] = torch.zeros((sample_len, s))
            attention_mask[csum:csum + s, csum:csum + s] = torch.ones((s, s))
        csum += s

    attention_mask = torch.zeros_like(attention_mask, dtype=torch.float).masked_fill_(
        ~attention_mask, float("-inf")
    )
    return attention_mask


def split_integer_exp_decay(S, ng_sample_decay=1.0):
    if ng_sample_decay == 1.0:
        N = random.randint(1, S)
    else:
        base = (1 - ng_sample_decay) / (1 - math.pow(ng_sample_decay, S))
        p = [base * math.pow(ng_sample_decay, i) for i in range(S)]
        N = random.choices(list(range(1, S + 1)), p, k=1)[0]
    cumsum = [0] + sorted(random.sample(range(1, S), N - 1)) + [S]
    result = [cumsum[i + 1] - cumsum[i] for i in range(len(cumsum) - 1)]
    return result, cumsum


def pil_img2rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGBA" or image.info.get("transparency", None) is not None:
        image = image.convert("RGBA")
        white = Image.new(mode="RGB", size=image.size, color=(255, 255, 255))
        white.paste(image, mask=image.split()[3])
        image = white
    else:
        image = image.convert("RGB")

    return image


def len2weight(x, loss_reduction='square'):
    if x == 0:
        return x
    if loss_reduction == 'token':
        return 1
    if loss_reduction == 'sample':
        return 1 / x
    if loss_reduction == 'square':
        return 1 / (x**0.5)
    raise NotImplementedError(loss_reduction)


def add_special_tokens(tokenizer):
    all_special_tokens = []
    for k, v in tokenizer.special_tokens_map.items():
        if isinstance(v, str):
            all_special_tokens.append(v)
        elif isinstance(v, list):
            all_special_tokens += v

    new_tokens = []

    if '<|im_start|>' not in all_special_tokens:
        new_tokens.append('<|im_start|>')

    if '<|im_end|>' not in all_special_tokens:
        new_tokens.append('<|im_end|>')

    if '<|vision_start|>' not in all_special_tokens:
        new_tokens.append('<|vision_start|>')

    if '<|vision_end|>' not in all_special_tokens:
        new_tokens.append('<|vision_end|>')

    num_new_tokens = tokenizer.add_tokens(new_tokens)

    # For tiktoken-based tokenizers (e.g. Kimi-VL TikTokenTokenizer),
    # add_tokens() only updates the HF wrapper but NOT the underlying
    # tiktoken Encoding, self.decoder, or self.encoder.  This causes
    # tokenizer.decode() to raise KeyError for newly-added token IDs.
    # Fix: rebuild the tiktoken Encoding with the new special tokens and
    # sync the decoder/encoder dicts so decode() works end-to-end.
    if num_new_tokens > 0 and hasattr(tokenizer, 'model') and hasattr(tokenizer, 'special_tokens'):
        try:
            from pathlib import Path

            import tiktoken
            from transformers.convert_slow_tokenizer import bytes_to_unicode

            # Collect newly added token → id mappings
            added_sp = dict(tokenizer.special_tokens)  # existing special tokens
            for tok_str in new_tokens:
                tok_id = tokenizer.convert_tokens_to_ids(tok_str)
                if tok_id not in added_sp.values():
                    added_sp[tok_str] = tok_id

            # Rebuild tiktoken Encoding with the expanded special_tokens
            tokenizer.special_tokens = added_sp
            tokenizer.model = tiktoken.Encoding(
                name=Path(tokenizer.vocab_file).name,
                pat_str=tokenizer.pat_str,
                mergeable_ranks=tiktoken.load.load_tiktoken_bpe(tokenizer.vocab_file),
                special_tokens=added_sp,
            )
            tokenizer.n_words = tokenizer.model.n_vocab

            # Sync decoder / encoder dicts
            byte_encoder = bytes_to_unicode()
            for tok_str, tok_id in added_sp.items():
                if tok_id not in tokenizer.decoder:
                    decoding = "".join(
                        byte_encoder[ord(c)] for c in
                        tokenizer.model.decode_single_token_bytes(tok_id).decode("latin-1")
                    )
                    tokenizer.decoder[tok_id] = decoding
                    tokenizer.encoder[decoding] = tok_id
        except Exception:
            pass  # non-tiktoken tokenizer, nothing to patch

    bos_token_id = tokenizer.convert_tokens_to_ids('<|im_start|>')
    eos_token_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    start_of_image = tokenizer.convert_tokens_to_ids('<|vision_start|>')
    end_of_image = tokenizer.convert_tokens_to_ids('<|vision_end|>')

    new_token_ids = dict(
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        start_of_image=start_of_image,
        end_of_image=end_of_image,
    )

    return tokenizer, new_token_ids, num_new_tokens


def build_flash_mask_index(sample_lens, split_lens, attn_modes, k_num_heads=1, device="cuda"):
    sample_len = sum(sample_lens)
    sample_end = 0
    sample_index = -1

    UTE = torch.arange(sample_len, device=device).to(torch.int32)
    LTS = torch.arange(sample_len, device=device).to(torch.int32)

    cur = 0
    for (l, attn_mode) in zip(split_lens, attn_modes):
        if cur >= sample_end:
            sample_index += 1
            sample_end += sample_lens[sample_index]

        next_cur = cur + l
        if attn_mode == "full":
            UTE[cur:next_cur] = cur
            LTS[cur:next_cur] = sample_end
        elif attn_mode == "noise":
            UTE[cur:next_cur] = cur
            # "gen latent is not attened by others"
            LTS[cur:next_cur] = next_cur
        else:
            LTS[cur:next_cur] = sample_end
        cur = next_cur

    UTE = UTE.reshape(1, 1, -1, 1)
    LTS = LTS.reshape(1, 1, -1, 1)
    # [batch_size, k_num_heads, k_seq_len, {1, 2, 4}].
    startend_row_indices = torch.cat([LTS, UTE], dim=-1).repeat(1, k_num_heads, 1, 1)
    return startend_row_indices
