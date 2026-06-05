from dataclasses import dataclass, field

from gpatch_v4.configs.utils import MappingProtocol


@dataclass
class GeminiSelfHealConfig(MappingProtocol):
    """Tuning parameters for GeminiNodeReplacer (used internally when enable_self_heal=True)."""

    recovery_max_wait: int = field(
        default=1800,
        metadata={"help": "Max seconds to wait for pod replacement."},
    )
    recovery_poll_interval: int = field(
        default=15,
        metadata={"help": "Seconds between recovery-ready polls."},
    )
    tag_recovery_timeout: int = field(
        default=300,
        metadata={"help": "Timeout param for tag_the_pods_to_recovery API."},
    )
    ray_port: int = field(
        default=6379,
        metadata={"help": "Ray head port for cluster rebuild."},
    )
