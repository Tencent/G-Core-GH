import sys
from pathlib import Path
from types import SimpleNamespace

import torch


MLITE_PATH = Path(__file__).resolve().parents[3] / "mlite" / "experimental" / "lite"
sys.path.insert(0, str(MLITE_PATH))

from gpatch_v4.training_backend.loss.mlite_specific_loss import (  # noqa: E402
    MliteFinetuneLossInput,
    mlite_cross_entropy_loss,
)


def test_mlite_dp_and_microbatch_scaling_matches_global_token_mean():
    global_valid_tokens = torch.tensor(4.0)
    dp_size = 2
    rank_microbatches = [
        [
            (torch.tensor([-1.0, -7.0]), torch.tensor([1.0, 0.0])),
            (torch.tensor([-2.0]), torch.tensor([1.0])),
        ],
        [
            (torch.tensor([-3.0]), torch.tensor([1.0])),
            (torch.tensor([-4.0]), torch.tensor([1.0])),
        ],
    ]

    rank_losses = []
    for microbatches in rank_microbatches:
        rank_loss = torch.zeros(())
        for log_probs, loss_mask in microbatches:
            result = mlite_cross_entropy_loss(
                SimpleNamespace(),
                MliteFinetuneLossInput(
                    log_probs=log_probs,
                    aligned_loss_mask=loss_mask,
                    global_valid_tokens=global_valid_tokens,
                    dp_size=dp_size,
                ),
            )
            rank_loss += result.loss
        rank_losses.append(rank_loss)

    ddp_averaged_loss = torch.stack(rank_losses).mean()
    direct_global_mean = torch.tensor((1.0 + 2.0 + 3.0 + 4.0) / 4.0)
    torch.testing.assert_close(ddp_averaged_loss, direct_global_mean)
