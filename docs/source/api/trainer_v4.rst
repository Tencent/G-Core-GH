Trainer V4
==========

Trainers
--------

Base Classes
^^^^^^^^^^^^

.. autoclass:: gpatch_v4.trainer.trainer_mixin.TrainerRetryMixin
   :members: launch_with_retry

.. autoclass:: gpatch_v4.trainer.base_trainer.BaseTrainer
   :members: init_distributed, build_model_and_optimizer, build_train_valid_test_data_iter, save_ckpt, load_ckpt, train_loop

GrpoTrainer
^^^^^^^^^^^

.. autoclass:: gpatch_v4.trainer.grpo_trainer.GrpoTrainer
   :members: __init__, launch, conv_mcore_to_hf, debug_update_weight

FinetuneTrainer
^^^^^^^^^^^^^^^

.. autoclass:: gpatch_v4.trainer.finetune_trainer.FinetuneTrainer
   :members: __init__, launch, conv_mcore_to_hf

DpoTrainer
^^^^^^^^^^

.. autoclass:: gpatch_v4.trainer.dpo_trainer.DpoTrainer
   :members: __init__, launch, conv_mcore_to_hf

OnPolicyDistillTrainer
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: gpatch_v4.trainer.on_policy_distill_trainer.OnPolicyDistillTrainer
   :members: __init__, launch, conv_mcore_to_hf, debug_update_weight

OffPolicyDistillTrainer
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: gpatch_v4.trainer.off_policy_distill_trainer.OffPolicyDistillTrainer
   :members: __init__, launch, conv_mcore_to_hf, test_ray_rpc

T2iGrpoTrainer
^^^^^^^^^^^^^^

.. autoclass:: gpatch_v4.trainer.t2i_grpo_trainer.T2iGrpoTrainer
   :members: __init__, launch

Trainer Helpers
^^^^^^^^^^^^^^^

.. automodule:: gpatch_v4.trainer.helper
   :members: convert_mcore_to_hf, get_nnodes, set_nnodes_default


Orchestration (``orches``)
--------------------------

Ray Lifecycle
^^^^^^^^^^^^^

.. automodule:: gpatch_v4.orches
   :members: init, shutdown, is_initialized, get

Placement Groups
^^^^^^^^^^^^^^^^

.. automodule:: gpatch_v4.orches.placement_group
   :members: create_placement_groups, create_train_group, create_sampler_group, create_bt_rm_group, create_gen_rm_group, create_infer_group, create_teacher_group, create_kv_store_group

.. autoclass:: gpatch_v4.orches.placement_group.InfoActor
   :members: get_ip_and_gpu_id

Ray Actor Groups
^^^^^^^^^^^^^^^^

.. autoclass:: gpatch_v4.orches.train_group.RayTrainGroup
   :members: __init__, init, setup_client, setup_rollout_generator, setup_model_and_optimizer, train_loop, update_weights, convert_to_hf_checkpoint, evaluate, check_actor_stuck, debug_update_weight

.. autoclass:: gpatch_v4.orches.sampler_group.RaySamplerGroup
   :members: __init__, get_sampler_engine_info, init, wake_up

.. autoclass:: gpatch_v4.orches.bt_rm_group.RayBtRmGroup
   :members: __init__, get_rm_engine_info, init, wake_up

.. autoclass:: gpatch_v4.orches.gen_rm_group.RayGenRmGroup
   :members: __init__, get_custom_actor_cls, get_rm_engine_info, init, wake_up

.. automodule:: gpatch_v4.orches.gen_rm_group
   :members: _allocate_rollout_engine_addr_and_ports_normal
   :noindex:

Base Actor
^^^^^^^^^^

.. autoclass:: gpatch_v4.orches.base_actor.RayBaseActor
   :members: get_master_addr_and_port


Reward
------

.. autoclass:: gpatch_v4.reward.RewardFactory
   :members: get_reward_engine

.. autoclass:: gpatch_v4.reward.base_reward.RewardAbc
   :members: setup_reward_model, compute_rewards

.. autoclass:: gpatch_v4.reward.rule_reward.RuleReward
   :members: setup_reward_model, compute_rewards


Rollout Generator
-----------------

.. autoclass:: gpatch_v4.rollout_generator.RolloutGeneratorFactory
   :members: get_rollout_generator

.. autoclass:: gpatch_v4.rollout_generator.generator_abc.RolloutGeneratorAbc
   :members: rollout_samples, generate_gen_rm_reward, calc_bt_rm_reward, __call__, clear_data_cache


Clients
-------

.. autoclass:: gpatch_v4.client.base_client.BaseClientAbc
   :members: get_engine_info, mark_ppo_step_begin, mark_ppo_step_end, wake_up, sleep

.. autoclass:: gpatch_v4.client.base_client.RmClientMixin
   :members: build_rpc_client

.. autoclass:: gpatch_v4.client.base_client.SamplerClientMixin
   :members: build_rpc_client

.. autoclass:: gpatch_v4.client.base_client.TeacherClientMixin
   :members: build_rpc_client
