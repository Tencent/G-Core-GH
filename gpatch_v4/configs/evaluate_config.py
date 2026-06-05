import collections
from dataclasses import asdict, dataclass, field
from typing import Any, List, Optional

from gpatch_v4.configs.utils import MappingProtocol


@dataclass
class EvaluateResultConfig(MappingProtocol):
    output_dir: Optional[str] = field(default=None, metadata={"help": "Output directory"})
    output_prefix: Optional[str] = field(default=None, metadata={"help": "Output prefix"})
    evaluate_py_path: Optional[str] = field(default=None, metadata={"help": "Evaluate python path"})
    evaluate_fn_name: Optional[str] = field(
        default=None, metadata={"help": "Evaluate function name"}
    )
