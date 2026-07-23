import socket
import time

import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from gpatch_v4.configs.config import (
    BasePolicyConfig,
    DpoConfig,
    EvaluateConfig,
    FinetuneConfig,
    InferenceConfig,
    OffPolicyDistillConfig,
    OnPolicyDistillConfig,
    RewardConfig,
    RlConfig,
    T2iRlConfig,
)
from gpatch_v4.orches.resource_allocator import (
    SHARED_WITH_POLICY,
    allocation_from_config,
)
from gpatch_v4.utils.common_utils import logging_rank0
from gpatch_v4.utils.placement import (
    is_partial_colocated,
    validate_partial_colocated_config,
)


@ray.remote(num_gpus=1)
class InfoActor:
    """Lightweight Ray actor used to discover node IP and GPU ID."""
    def get_ip_and_gpu_id(self):
        """Return ``(node_ip, gpu_id)``.

        Returns
        -------
        tuple[str, int]
        """
        return ray.util.get_node_ip_address(), ray.get_gpu_ids()[0]


def sort_key(x):
    """Sort key for ordering placement group bundles by node IP and GPU ID.

    Parameters
    ----------
    x : tuple[int, str, int]
        ``(index, node_identifier, gpu_id)``.

    Returns
    -------
    tuple
    """
    index, node_identifier, gpu_id = x
    # Sort by node IP number and then by GPU ID
    try:
        # try to parse it as an IP address.
        ip_address = node_identifier
        node_ip_parts = list(map(int, ip_address.split(".")))
    except ValueError:
        # Try to resolve the hostname to an IP address.
        try:
            ip_address = socket.gethostbyname(node_identifier)
            node_ip_parts = list(map(int, ip_address.split(".")))
        except (socket.gaierror, TypeError):
            # Instead, we convert each character of the original identifier string
            # to its ASCII value. This provides a stable and consistent numerical
            # representation that allows for sorting.
            node_ip_parts = [ord(c) for c in node_identifier]

    return (node_ip_parts, gpu_id)


def compute_gen_rm_placement(total_gpus, rm_mp_sizes, manual_allocations=None):
    """Compute GPU allocation per RM group.

    RMs with a manual allocation keep that exact GPU count.  Any
    remaining GPUs are distributed to the other RMs via the legacy
    round-robin algorithm.

    Parameters
    ----------
    total_gpus : int
    rm_mp_sizes : list of tuple[int, int]
        ``(rm_idx, mp_size)`` pairs.
    manual_allocations : dict[int, int], optional
        Exact ``rm_idx -> allocated_total_gpus`` overrides.

    Returns
    -------
    dict[int, int]
        ``rm_idx -> allocated_total_gpus``.
    """
    manual_allocations = manual_allocations or {}
    rm_mp_by_idx = {}
    for rm_idx, mp in rm_mp_sizes:
        assert rm_idx not in rm_mp_by_idx, f"duplicate gen-rm index {rm_idx}"
        rm_mp_by_idx[rm_idx] = mp

    allocations = {rm_idx: 0 for rm_idx, _ in rm_mp_sizes}
    for rm_idx, allocated_gpus in manual_allocations.items():
        assert rm_idx in rm_mp_by_idx, f"manual allocation references unknown gen-rm {rm_idx}"
        assert isinstance(allocated_gpus, int) and not isinstance(allocated_gpus, bool), (
            f"allocated_gpus for gen-rm {rm_idx} must be a positive integer, "
            f"got {allocated_gpus!r}"
        )
        assert allocated_gpus > 0, (
            f"allocated_gpus for gen-rm {rm_idx} must be positive, got {allocated_gpus}"
        )
        mp = rm_mp_by_idx[rm_idx]
        assert allocated_gpus >= mp, (
            f"allocated_gpus({allocated_gpus}) must be >= mp_size({mp}) for gen-rm {rm_idx}"
        )
        assert allocated_gpus % mp == 0, (
            f"allocated_gpus({allocated_gpus}) must be divisible by mp_size({mp}) "
            f"for gen-rm {rm_idx}"
        )
        allocations[rm_idx] = allocated_gpus

    manual_total = sum(manual_allocations.values())
    assert manual_total <= total_gpus, (
        f"Manual gen-rm allocations require {manual_total} GPUs, got budget {total_gpus}"
    )

    auto_rm_mp_sizes = [
        (rm_idx, mp) for rm_idx, mp in rm_mp_sizes if rm_idx not in manual_allocations
    ]
    auto_min = sum(mp for _, mp in auto_rm_mp_sizes)
    remaining = total_gpus - manual_total
    assert remaining >= auto_min, (
        f"Need at least {auto_min} GPUs for non-manual gen-rms after manual allocations, "
        f"got {remaining}"
    )

    sorted_rm_mp_sizes = sorted(auto_rm_mp_sizes, key=lambda x: x[1], reverse=True)

    while remaining > 0:
        allocated_any = False
        for rm_idx, mp in sorted_rm_mp_sizes:
            if remaining >= mp:
                allocations[rm_idx] += mp
                remaining -= mp
                allocated_any = True
        if not allocated_any:
            break

    return allocations


def compute_gen_rm_config_placement(gen_rm_config, total_gpus):
    """Compute per-RM GPU allocation directly from ``gen_rm`` config."""
    reward_model_info = gen_rm_config.reward_model_info
    infer_engine_configs = gen_rm_config.infer_engine_configs
    assert reward_model_info is not None, "gen_rm.reward_model_info must be configured"
    assert infer_engine_configs is not None, "gen_rm.infer_engine_configs must be configured"
    assert len(reward_model_info) == len(infer_engine_configs), (
        "gen_rm.reward_model_info and gen_rm.infer_engine_configs must be 1:1, "
        f"got {len(reward_model_info)} reward_model_info entries and "
        f"{len(infer_engine_configs)} infer_engine_configs"
    )

    rm_mp_sizes = []
    manual_allocations = {}
    for rm_idx, ie_cfg in enumerate(infer_engine_configs):
        dc = ie_cfg.dist_config
        mp = dc.tensor_model_parallel_size * dc.pipeline_model_parallel_size
        rm_mp_sizes.append((rm_idx, mp))
        allocated_gpus = getattr(ie_cfg, "allocated_gpus", None)
        if allocated_gpus is not None:
            manual_allocations[rm_idx] = allocated_gpus

    return compute_gen_rm_placement(total_gpus, rm_mp_sizes, manual_allocations or None)


def _create_placement_group(num_gpus):
    """Create a Ray placement group and discover GPU assignments.

    Parameters
    ----------
    num_gpus : int

    Returns
    -------
    tuple[PlacementGroup, list[int]]
        ``(pg, reordered_bundle_indices)`` sorted by node IP and GPU ID.
    """
    bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
    pg = placement_group(bundles, strategy="PACK")
    num_bundles = len(bundles)

    ray.get(pg.ready())
    # use info actor to get the GPU id
    info_actors = []
    for i in range(num_bundles):
        info_actors.append(
            InfoActor.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=i,
                )
            ).remote()
        )
    gpu_ids = ray.get([actor.get_ip_and_gpu_id.remote() for actor in info_actors])
    for actor in info_actors:
        ray.kill(actor)

    bundle_infos = [(i, gpu_ids[i][0], gpu_ids[i][1]) for i in range(num_bundles)]
    pg_reordered_bundle_indices = [
        bundle_info[0] for bundle_info in sorted(bundle_infos, key=sort_key)
    ]
    for i in range(num_bundles):
        actual_bundle_index = pg_reordered_bundle_indices[i]
        print(
            f"  bundle {i:4}, actual_bundle_index: {actual_bundle_index:4}, "
            f"node: {gpu_ids[actual_bundle_index][0]}, gpu: {gpu_ids[actual_bundle_index][1]}"
        )

    return pg, pg_reordered_bundle_indices


def _remove_placement_group(pg):
    """Remove a placement group and wait briefly for cleanup.

    Parameters
    ----------
    pg : PlacementGroup
    """
    remove_placement_group(pg)
    # Wait until placement group is killed.
    time.sleep(1)


def create_placement_groups(config):
    """Create Ray placement groups based on the training configuration.

    Layout is driven by a
    :class:`~gpatch_v4.orches.resource_allocator.ResourceAllocation`
    derived from *config* via :func:`allocation_from_config` (single
    source of truth for per-role resource intent). Sharing relations
    (colocate / disaggregated / ``SHARED_WITH_POLICY``) are applied
    here on top of the allocation.

    Parameters
    ----------
    config : object
        ``RlConfig``, ``T2iRlConfig``, etc.

    Returns
    -------
    dict[str, tuple[PlacementGroup, list[int]]]
        Role name -> ``(pg, reordered_bundle_indices)``.
    """
    allocation = allocation_from_config(config)

    if isinstance(config, InferenceConfig):
        sampler_num_gpus = allocation.role_num_gpus("sampler")
        logging_rank0(f"creating placement groups with {sampler_num_gpus} ...")
        pg, pg_reordered_bundle_indices = _create_placement_group(sampler_num_gpus)

        groups = {"sampler": (pg, pg_reordered_bundle_indices[:sampler_num_gpus])}
        return groups

    #TODO: 将这些按照 config 类型的改写成 config training 里的字段
    if isinstance(
        config, (
            EvaluateConfig, FinetuneConfig, RlConfig, T2iRlConfig, OnPolicyDistillConfig,
            OffPolicyDistillConfig, DpoConfig, RewardConfig
        )
    ):
        # ``kv`` / ``teacher_*`` slice widths intentionally use the **policy**
        # ``num_gpus_per_node`` (preserved from pre-refactor behavior), even
        # when those roles declare their own ``num_gpus_per_node`` in their
        # ``dist_config``.  Fixing this is out of scope for this PR.
        policy_gpus_per_node = allocation.role_gpus_per_node["policy"]

        teacher_roles = [
            r for r in allocation.role_nnodes if r == "teacher" or r.startswith("teacher_")
        ]
        total_teacher_nnodes = sum(allocation.role_nnodes[r] for r in teacher_roles)

        kv_nnodes = allocation.role_nnodes.get("kv", 0)

        if config.placement_type == "colocate":
            # All teachers colocate with student — share policy GPUs (time-shared via offloading)
            policy_nnodes = allocation.role_nnodes["policy"]
            assert total_teacher_nnodes <= policy_nnodes, (
                f"Total teacher nodes ({total_teacher_nnodes}) exceeds "
                f"policy nodes ({policy_nnodes})"
            )
            policy_num_gpus = allocation.role_num_gpus("policy")
            sampler_num_gpus = policy_num_gpus  # colocate: all roles share policy GPUs
            gen_rm_num_gpus = policy_num_gpus
            bt_rm_num_gpus = policy_num_gpus
            teacher_num_gpus = policy_num_gpus
            num_gpus = policy_num_gpus
            rollout_offset = 0
            gen_rm_offset = 0
            bt_rm_offset = 0
            teacher_offset = 0
        elif is_partial_colocated(config):
            validate_partial_colocated_config(config)
            assert isinstance(config, RlConfig
                             ), ("partial_colocated placement currently supports only RlConfig")
            policy_num_gpus = allocation.role_num_gpus("policy")
            sampler_num_gpus = allocation.role_num_gpus("sampler")
            gen_rm_num_gpus = allocation.role_num_gpus("gen_rm")
            assert policy_num_gpus == sampler_num_gpus + gen_rm_num_gpus, (
                "partial_colocated requires policy GPU count to equal "
                f"sampler + gen_rm, got {policy_num_gpus} != "
                f"{sampler_num_gpus} + {gen_rm_num_gpus}"
            )
            bt_rm_num_gpus = 0
            teacher_num_gpus = 0

            num_gpus = policy_num_gpus
            rollout_offset = 0
            gen_rm_offset = sampler_num_gpus
            bt_rm_offset = policy_num_gpus
            teacher_offset = policy_num_gpus
        elif config.placement_type == "disaggregated":
            # Each role may specify its own num_gpus_per_node for sub-node allocation.
            # For example, sampler and gen_rm can each use 4 GPUs on a single 8-GPU node.
            policy_num_gpus = allocation.role_num_gpus("policy")
            sampler_num_gpus = allocation.role_num_gpus("sampler")
            gen_rm_num_gpus = allocation.role_num_gpus("gen_rm")
            bt_rm_num_gpus = allocation.role_num_gpus("bt_rm")
            # Teachers intentionally use policy_gpus_per_node (see comment above).
            teacher_num_gpus = total_teacher_nnodes * policy_gpus_per_node

            num_gpus = (
                policy_num_gpus + sampler_num_gpus + gen_rm_num_gpus + bt_rm_num_gpus +
                teacher_num_gpus
            )
            rollout_offset = policy_num_gpus
            gen_rm_offset = rollout_offset + sampler_num_gpus
            bt_rm_offset = gen_rm_offset + gen_rm_num_gpus
            teacher_offset = bt_rm_offset + bt_rm_num_gpus
        else:
            raise NotImplementedError(f"error")

        logging_rank0(f"creating placement groups with {num_gpus} ...")
        pg, pg_reordered_bundle_indices = _create_placement_group(num_gpus)

        # ``kv`` / ``training_plt`` are SHARED_WITH_POLICY and always slice
        # from the pg prefix; they are not counted in ``num_gpus``.
        kv_offset = 0
        groups = {
            "policy": (pg, pg_reordered_bundle_indices[:policy_num_gpus]),
            "gen_rm":
                (pg, pg_reordered_bundle_indices[gen_rm_offset:gen_rm_offset + gen_rm_num_gpus]),
            "bt_rm": (pg, pg_reordered_bundle_indices[bt_rm_offset:bt_rm_offset + bt_rm_num_gpus]),
            "sampler":
                (pg, pg_reordered_bundle_indices[rollout_offset:rollout_offset + sampler_num_gpus]),
            'kv':
                (
                    pg, pg_reordered_bundle_indices[kv_offset:kv_offset +
                                                    kv_nnodes * policy_gpus_per_node]
                ),
            'training_plt': (pg, pg_reordered_bundle_indices[0:1]),
        }
        if isinstance(config, (OffPolicyDistillConfig)):
            groups["teacher"] = (
                pg, pg_reordered_bundle_indices[teacher_offset:teacher_offset + teacher_num_gpus]
            )

        if isinstance(config, (OnPolicyDistillConfig)):
            # Teacher placement: teacher 之间隔离摆放
            curr_teacher_offset = teacher_offset
            for t_name in config.teachers.keys():
                t_role = f"teacher_{t_name}"
                t_nn = allocation.role_nnodes[t_role]
                t_slice = t_nn * policy_gpus_per_node
                groups[t_role] = (
                    pg,
                    pg_reordered_bundle_indices[curr_teacher_offset:curr_teacher_offset + t_slice]
                )
                curr_teacher_offset += t_slice

        return groups
    else:
        raise NotImplementedError(f'unknown config {config}')


def create_train_group(config, pgs):
    """Create a ``RayTrainGroup`` for the policy role.

    Parameters
    ----------
    config : object
    pgs : dict
        Placement groups returned by ``create_placement_groups``.

    Returns
    -------
    RayTrainGroup
    """
    from gpatch_v4.orches.train_group import RayTrainGroup
    return RayTrainGroup(
        config=config,
        num_nodes=config.policy.dist_config.nnodes,
        num_gpus_per_node=config.policy.dist_config.num_gpus_per_node,
        pg=pgs['policy'],
        role="policy",
    )


def create_gen_rm_group(config, pgs):
    """Create ``RayGenRmGroup`` instances for the gen-RM role.

    Always returns a **list** of groups — one per RM. Each group
    receives only the GPU bundles allocated to that RM via
    ``compute_gen_rm_placement``.

    Parameters
    ----------
    config : object
    pgs : dict

    Returns
    -------
    list[RayGenRmGroup]
        One group per reward model.
    """
    from gpatch_v4.orches.gen_rm_group import RayGenRmGroup

    if getattr(config.gen_rm, "destroy_engine_after_generation", False):
        assert isinstance(
            config, RlConfig
        ), ("gen_rm.destroy_engine_after_generation currently only supports LLM GRPO RlConfig")
        assert config.gen_rm.backend == "sglang", (
            "gen_rm.destroy_engine_after_generation only supports sglang backend"
        )

    pg, gen_rm_bundle_indices = pgs['gen_rm']
    total_gpus = len(gen_rm_bundle_indices)
    allocations = compute_gen_rm_config_placement(config.gen_rm, total_gpus)
    num_rms = len(config.gen_rm.reward_model_info)

    groups = []
    cursor = 0
    for rm_idx in range(num_rms):
        n = allocations[rm_idx]
        rm_bundles = gen_rm_bundle_indices[cursor:cursor + n]
        cursor += n
        groups.append(
            RayGenRmGroup(
                config=config,
                pg=(pg, rm_bundles),
                rm_idx=rm_idx,
                allocated_num_gpus=n,
                role="gen_rm",
            )
        )
    return groups


def create_bt_rm_group(config, pgs):
    """Create a ``RayBtRmGroup`` for the BT reward model role.

    Parameters
    ----------
    config : object
    pgs : dict

    Returns
    -------
    RayBtRmGroup
    """
    from gpatch_v4.orches.bt_rm_group import RayBtRmGroup
    return RayBtRmGroup(
        config=config,
        num_nodes=config.bt_rm.dist_config.nnodes,
        num_gpus_per_node=config.bt_rm.dist_config.num_gpus_per_node,
        pg=pgs['bt_rm'],
        role="bt_rm",
    )


def create_sampler_group(config, pgs):
    """Create a ``RaySamplerGroup`` for the sampler role.

    Parameters
    ----------
    config : object
    pgs : dict

    Returns
    -------
    RaySamplerGroup
    """
    from gpatch_v4.orches.sampler_group import RaySamplerGroup
    return RaySamplerGroup(
        config=config,
        num_nodes=config.sampler.dist_config.nnodes,
        num_gpus_per_node=config.sampler.dist_config.num_gpus_per_node,
        pg=pgs['sampler'],
        role='sampler',
    )


def create_infer_group(config, pgs):
    """Create a ``RaySamplerGroup`` for inference-only mode.

    Parameters
    ----------
    config : object
    pgs : dict

    Returns
    -------
    RaySamplerGroup
    """
    from gpatch_v4.orches.sampler_group import RaySamplerGroup
    return RaySamplerGroup(
        config=config,
        num_nodes=config.sampler.dist_config.nnodes,
        num_gpus_per_node=config.sampler.dist_config.num_gpus_per_node,
        pg=pgs['sampler'],
        role='sampler',
    )


def create_teacher_group(config, pgs):
    """Create a single ``RayTeacherGroup`` for ``OffPolicyDistillConfig``.

    Parameters
    ----------
    config : OffPolicyDistillConfig
        Has singular ``teacher`` field.
    pgs : dict

    Returns
    -------
    RayTeacherGroup
    """
    from gpatch_v4.orches.teacher_group import RayTeacherGroup
    return RayTeacherGroup(
        config=config,
        num_nodes=config.teacher.dist_config.nnodes,
        num_gpus_per_node=config.policy.dist_config.num_gpus_per_node,
        pg=pgs['teacher'],
        role='teacher',
    )


def create_teacher_groups(config, pgs):
    """Create ``RayTeacherGroup`` instances for all configured teachers.

    For each teacher in ``config.teachers``, a shallow copy of *config*
    is made with ``config.teacher`` set to that teacher's config so the
    existing ``RayTeacherGroup`` / ``DistillTeacherActor`` code works
    unchanged.

    Parameters
    ----------
    config : OnPolicyDistillConfig
    pgs : dict

    Returns
    -------
    dict[str, RayTeacherGroup]
        Teacher name -> group.
    """
    import copy

    from gpatch_v4.orches.teacher_group import RayTeacherGroup

    groups = {}
    for t_name, t_cfg in config.teachers.items():
        config_copy = copy.copy(config)
        t_cfg_ = BasePolicyConfig(**t_cfg)

        t_nnodes = t_cfg_.dist_config.nnodes
        ngpu_per_node = t_cfg_.dist_config.num_gpus_per_node
        config_copy.teacher = t_cfg_

        pg_key = f"teacher_{t_name}"
        groups[t_name] = RayTeacherGroup(
            config=config_copy,
            num_nodes=t_nnodes,
            num_gpus_per_node=ngpu_per_node,
            pg=pgs[pg_key],
            role=pg_key,
        )

    return groups


def create_kv_store_group(config, pgs):
    """Create a ``RayKvStoreGroup`` for the key-value store role.

    Parameters
    ----------
    config : object
    pgs : dict

    Returns
    -------
    RayKvStoreGroup
    """
    from gpatch_v4.orches.kv_store_group import RayKvStoreGroup
    return RayKvStoreGroup(
        config=config,
        num_nodes=config.kv.dist_config.nnodes,
        num_gpus_per_node=config.kv.dist_config.num_gpus_per_node,
        pg=pgs['kv'],
        role="kv",
    )


def create_training_plt_group(config, pgs):
    """Create a ``RayTrainingPltGroup`` for training.

    Parameters
    ----------
    config : object
    pgs : dict

    Returns
    -------
    RayTrainingPltGroup
    """
    from gpatch_v4.orches.training_plt_group import RayTrainingPltGroup
    return RayTrainingPltGroup(
        config=config,
        num_nodes=1,
        num_gpus_per_node=1,
        pg=pgs['training_plt'],
        role="training_plt",  # hardcode, nerver change
    )


def create_rollout_controller(config):
    """Create a ``RolloutController`` Ray actor for centralized rollout management.

    The controller runs with 0 GPUs and 1 CPU. It owns the data source,
    sampler client, gen-RM client, and BT-RM client, performing all
    rollout I/O in a single process before splitting results to DP ranks.

    Parameters
    ----------
    config : RlConfig

    Returns
    -------
    ray.actor.ActorHandle
    """
    from gpatch_v4.rollout_generator.async_rollout.rollout_controller import (
        RolloutController,
    )

    ControllerActor = ray.remote(num_cpus=1, num_gpus=0)(RolloutController)
    controller = ControllerActor.options(name="rollout_controller", ).remote(config)
    return controller
