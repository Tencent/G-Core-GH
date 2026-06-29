from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

from gpatch_v4.configs.utils import MappingProtocol


@dataclass
class LLMProxyConfig(MappingProtocol):
    proxy_type: str = field(
        default="engine", metadata={"help": "llm proxy type: [engine, random]."}
    )
    proxy_config: Dict = field(default_factory=dict, metadata={"help": "llm proxy config."})


@dataclass
class EnvManagerConfig(MappingProtocol):
    llm_proxy: LLMProxyConfig = field(
        default_factory=LLMProxyConfig, metadata={"help": "llm proxy config."}
    )
    max_traj_per_env: int = field(default=16, metadata={"help": "max_traj_per_env"})
    group_per_worker: int = field(default=4, metadata={"help": "group_per_worker"})
    group_replicate: int = field(
        default=1, metadata={"help": "number of times to replicate each group"}
    )


@dataclass
class EnvToolConfig(MappingProtocol):
    use_tools: bool = field(default=False, metadata={"help": "use tools"})
    tools_json_path: str = field(default="", metadata={"help": "path to tools json file"})
    tool_call_parser: str = field(default="", metadata={"help": "tool call parser name"})


@dataclass
class EnvTemplateConfig(MappingProtocol):
    env_type: str = field(default="sokoban", metadata={"help": "env_type"})
    custom_env_cls: str = field(
        default="",
        metadata={
            "help":
                (
                    "When set, register this class under env_cfg_template.env_type for gem.make, "
                    "e.g. 'my_pkg.my_env:MyEnv' (same format as gem.register entry_point)."
                )
        },
    )
    max_steps: int = field(default=-1, metadata={"help": "max_steps"})
    max_tokens_per_step: Optional[int] = field(
        default=None, metadata={"help": "max_tokens_per_step"}
    )
    env_manager_cls: str = field(
        default=
        "gpatch_v4.agentic.env_manager.step_vl_traj_env_manager_sokoban.StepVLTrajEnvManager",
        metadata={"help": ""}
    )
    use_thread_lock: bool = field(default=True, metadata={"help": ""})
    agent_system_template: str = field(default="", metadata={"help": ""})
    agent_template: str = field(
        default="",
        metadata={"help": "Template for user message (observation, turn_idx, suffix, etc.)."}
    )
    pre_step_template: str = field(default="", metadata={"help": ""})
    next_step_template: str = field(default="", metadata={"help": ""})
    training_id: str = field(default="", metadata={"help": "training run identifier"})
    env_config: Dict = field(default_factory=dict, metadata={"help": "llm proxy config."})
    history_length: int = field(default=5, metadata={"help": ""})
    env_tool_config: EnvToolConfig = field(default_factory=EnvToolConfig)


@dataclass
class RewardNormalizationConfig(MappingProtocol):
    grouping: str = field(default="state", metadata={"help": "state / batch / inductive"})
    method: str = field(default="identity", metadata={"help": "asym_clip / identity / mean_std"})


@dataclass
class AgenticConfig(MappingProtocol):
    max_concurrency: int = field(
        default=0,
        metadata={
            "help":
                (
                    "Upper bound on concurrent ``TrajEnvManager.run`` calls across all "
                    "``EnvAgentLoopActor`` workers. Each actor uses "
                    "``max(1, max_concurrency // num_agent_loop_workers)`` slots. "
                    "0 disables limiting (only ``asyncio`` default thread pool applies)."
                )
        },
    )
    env_cfg_template: EnvTemplateConfig = field(default_factory=EnvTemplateConfig)
    train_env_manager: EnvManagerConfig = field(default_factory=EnvManagerConfig)
    step_reward_gamma: float = field(
        default=0.95, metadata={"help": "Gamma parameter for step reward calculation"}
    )
    batch_adjust_mode: str = field(
        default="copy", metadata={"help": "batch adjust mode: copy or delete"}
    )
    reward_normalization: RewardNormalizationConfig = field(
        default_factory=RewardNormalizationConfig
    )
