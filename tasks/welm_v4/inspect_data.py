import torch

a = torch.load("debug/debug_batch_dp0_tp0_cp0.pt")
b = torch.load("debug/debug_batch_dp0_tp0_cp1.pt")

input_ids_ngram_origin = a['input_ids_ngram_origin'].cpu()


def test(input1, input2, seq_dim=2, cp_size=2):
    input1_ = input1.view(
        *input1.shape[0:seq_dim],
        2,
        input1.shape[seq_dim] // 2,
        *input1.shape[(seq_dim + 1):],
    )
    input2_ = input2.view(
        *input2.shape[0:seq_dim],
        2,
        input2.shape[seq_dim] // 2,
        *input2.shape[(seq_dim + 1):],
    )
    gathered_logits = [input1_, input2_]

    reorded_logits = [None for _ in range(2 * cp_size)]
    if seq_dim == 1:
        for rank in range(cp_size):
            reorded_logits[rank] = gathered_logits[rank][:, 0]
            reorded_logits[2 * cp_size - rank - 1] = gathered_logits[rank][:, 1]
    elif seq_dim == 2:
        for rank in range(cp_size):
            reorded_logits[rank] = gathered_logits[rank][:, :, 0]
            reorded_logits[2 * cp_size - rank - 1] = gathered_logits[rank][:, :, 1]
    else:
        assert False

    gathered_logits = torch.cat(reorded_logits, dim=seq_dim)
    return gathered_logits


input1 = a['input_ids_ngram'].cpu()
input2 = b['input_ids_ngram'].cpu()

resort_data = test(input1, input2, seq_dim=2, cp_size=2)

print(torch.all(resort_data == input_ids_ngram_origin))
