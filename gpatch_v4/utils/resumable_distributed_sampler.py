from torch.utils.data.distributed import DistributedSampler


class ResumableDistributedSampler(DistributedSampler):
    """DistributedSampler that can resume from a given training step.

    Parameters
    ----------
    args : Dataset
    **kwargs
    """
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.start_index = 0

    def set_start_index(self, start_step: int, batch_size: int):
        """Set the starting index for resuming iteration.

        Parameters
        ----------
        start_step : int
        batch_size : int
        """
        samples_per_epoch = self.num_samples
        steps_per_epoch = samples_per_epoch // batch_size

        if steps_per_epoch > 0:
            completed_epochs = start_step // steps_per_epoch
            self.set_epoch(completed_epochs)
            self.start_index = (start_step % steps_per_epoch) * batch_size
        else:
            self.start_index = 0

    def __iter__(self):
        indices = list(super().__iter__())
        return iter(indices[self.start_index:])

    def __len__(self):
        return self.num_samples - self.start_index
