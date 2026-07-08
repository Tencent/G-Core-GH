from typing_extensions import override

from gpatch_v4.actor.finetune_actor import FinetuneActor


class RewardActor(FinetuneActor):
    """Ray actor for Bradley-Terry reward-model training (output_scalar).

    Reuses the SFT training loop: data is pairwise (chosen|rejected), the model
    carries a scalar reward head, and the ``rm_bt`` loss ranks chosen vs
    rejected. No reference model is needed.
    """

    train_log_tag = "RM"

    @override
    def validated_config(self):
        super().validated_config()
        assert self.config.training.loss_func == "rm_bt", \
            f"RewardActor requires loss_func='rm_bt', got {self.config.training.loss_func}"
