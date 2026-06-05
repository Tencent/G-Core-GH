import torch

from tasks.t2i_dpo.oteam_44.data import build_dataloader

from gpatch_v4.trainer.t2i_dpo_trainer import T2iDpoTrainer


class OteamDpoTrainer(T2iDpoTrainer):
    def build_train_valid_test_data_iter(self):
        self.args.training.train_method = 'dpo'
        return build_dataloader(self.args, tokenizer=None, image_processor=self.image_processor)
