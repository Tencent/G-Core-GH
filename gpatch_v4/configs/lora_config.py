from dataclasses import dataclass, field
from typing import List, Optional

from gpatch_v4.configs.utils import MappingProtocol


@dataclass
class LoRAConfig(MappingProtocol):
    """LoRA / PEFT settings under ``policy.lora``.

    Attributes
    ----------
    rank : int
        LoRA rank; ``0`` disables PEFT (full fine-tuning).
    alpha : int
        LoRA scaling factor.
    type : str
        ``lora``, ``canonical_lora``
    dropout : float
    dropout_position : str
        ``pre`` or ``post``.
    target_modules : list of str or None
        Module name patterns; defaults to attention + MLP linears.
    exclude_modules : list of str
    lora_A_init_method : str
    lora_B_init_method : str
    a2a_experimental : bool
    dtype : str or None
        Optional adapter dtype name (reserved).
    """

    rank: int = field(default=0, metadata={"help": "LoRA rank; 0 disables PEFT"})
    alpha: int = field(default=32, metadata={"help": "LoRA alpha"})
    type: str = field(default="canonical_lora", metadata={"help": "PEFT type"})
    dropout: float = field(default=0.0, metadata={"help": "LoRA dropout"})
    dropout_position: str = field(default="pre", metadata={"help": "LoRA dropout position"})
    target_modules: Optional[List[str]] = field(
        default=None, metadata={"help": "Target module name patterns"}
    )
    exclude_modules: List[str] = field(
        default_factory=list, metadata={"help": "Excluded module name patterns"}
    )
    lora_A_init_method: str = field(default="xavier", metadata={"help": "LoRA A init"})
    lora_B_init_method: str = field(default="zero", metadata={"help": "LoRA B init"})
    a2a_experimental: bool = field(default=False, metadata={"help": "A2A experimental comm"})
    dtype: Optional[str] = field(default=None, metadata={"help": "Adapter dtype (optional)"})
    check_lora_all_coverage: bool = field(
        default=True,
        metadata={"help": "Check LoRA coverage(not including output_layer/moe router)"}
    )
    verify_weight_consistency: bool = field(
        default=True,
        metadata={"help": "Verify LoRA weight consistency across TP/DP/CP ranks at init"}
    )

    def __post_init__(self):
        # DoRA 没测试过，先不使用，mbridge 也没有移过来
        assert self.type in ["lora", "canonical_lora"]

    def enabled(self) -> bool:
        """Return whether PEFT is active."""
        return self.rank > 0
