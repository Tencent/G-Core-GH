from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    """Abstract base class for all trainers.

    Parameters
    ----------
    args : object
    """
    def __init__(self, args):
        self.args = args

    @abstractmethod
    def init_distributed(self):
        """Initialize distributed training environment."""
        ...

    @abstractmethod
    def build_model_and_optimizer(self):
        """Build the model and optimizer."""
        ...

    @abstractmethod
    def build_train_valid_test_data_iter(self):
        """Build data iterators for train, validation, and test splits."""
        ...

    @abstractmethod
    def save_ckpt(self):
        """Save a training checkpoint."""
        ...

    @abstractmethod
    def load_ckpt(self):
        """Load a training checkpoint."""
        ...

    @abstractmethod
    def train_loop(self):
        """Execute the main training loop."""
        ...
