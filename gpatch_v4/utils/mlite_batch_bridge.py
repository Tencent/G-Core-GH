from typing import List

import torch
from tensordict import TensorDict


def build_mlite_finetune_source_batch(aligned_loss_masks: List[torch.Tensor], ) -> TensorDict:
    return TensorDict(
        {
            "aligned_loss_mask":
                torch.nested.as_nested_tensor(
                    aligned_loss_masks,
                    layout=torch.jagged,
                )
        },
        batch_size=[len(aligned_loss_masks)],
    )
