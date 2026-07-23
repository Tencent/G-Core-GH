import torch

#TODO: 将 samrt pad 的file 搬到 v4 里，拿掉非 v4 的依赖
from gpatch.core.smart_pad_helper import postprocess_packed_seqs, preprocess_packed_seqs


def gptmodel_pack_foward(model, batch, fwd_kwargs, config):
    # If a pre-built packed_seq_params is already present (e.g. DSv4 mcore CP THD
    # built by _sft_train_mcore_cp_thd), pass it through directly without
    # reconstructing a new PSP from the pad-mask.  The model receives the correct
    # qkv_format='thd' + cp_partition_mode='contiguous' PSP.
    if fwd_kwargs.get("packed_seq_params", None) is not None:
        return model(
            input_ids=fwd_kwargs["input_ids"],
            position_ids=fwd_kwargs.get("position_ids"),
            attention_mask=None,
            labels=None,
            packed_seq_params=fwd_kwargs["packed_seq_params"],
        )

    #TODO(hessianiu)
    # add support for contiguous cp_partition_mode
    assert config.debug.debug_align_mode, (
        "If you do not provide packed_seq_params, debug.debug_align_mode must be True"
    )
    assert not config.training.enable_mtp, ("internal packing path does not support MTP training")

    # pack seq len
    assert not (
        model.position_embedding_type == 'mrope' and not model.config.multi_latent_attention
    )
    # [b, s] tensor indicating pad (0) or not (1)
    cur_mbs, cur_max_seqlen = fwd_kwargs['input_ids'].shape[:2]
    cur_actual_seqlen = batch['sequence_lengths'].unsqueeze(1).expand(-1, cur_max_seqlen)
    tmpa = torch.arange(cur_max_seqlen, device='cuda',
                        dtype=torch.int32).unsqueeze(0).expand(cur_mbs, -1)
    pad_mask = torch.ones(cur_mbs, cur_max_seqlen, device='cuda', dtype=torch.bool)
    pad_mask[tmpa >= cur_actual_seqlen] = False
    input_ids_rmpad, packed_seq_params = preprocess_packed_seqs(
        fwd_kwargs['input_ids'],
        pad_mask,
        pre_process=model.pre_process,
    )
    input_ids_rmpad = input_ids_rmpad.contiguous()
    output_rmpad = model(
        input_ids=input_ids_rmpad,
        position_ids=None,
        attention_mask=None,
        labels=None,
        packed_seq_params=packed_seq_params,
    )
    parallel_logits = postprocess_packed_seqs(
        output_rmpad,
        packed_seq_params,
        pad_mask,
        cur_mbs,
        cur_max_seqlen,
        post_process=model.post_process,
    )
    return parallel_logits
