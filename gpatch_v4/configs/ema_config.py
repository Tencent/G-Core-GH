from dataclasses import asdict, dataclass, field
from typing import Any, List, Optional

from gpatch_v4.configs.utils import MappingProtocol


@dataclass
class EmaConfig(MappingProtocol):
    """Configuration for Exponential Moving Average (EMA) of model weights.

    Attributes
    ----------
    use_ema : bool
    ema_decay : float
        Closer to 1.0 → slower averaging.
    ema_start_step : int
    """
    use_ema: bool = field(default=False, metadata={"help": "Whether to use ema."})
    ema_decay: float = field(
        default=0.995,
        metadata={"help": "Decay rate for the EMA."},
    )
    ema_start_step: int = field(
        default=0,
        metadata={"help": "Start step for the EMA."},
    )
