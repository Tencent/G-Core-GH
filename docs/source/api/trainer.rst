Trainer
=======


.. automodule:: gpatch.training.v3.ppo_actor
   :members: train_ppo_actor_v3

.. automodule:: gpatch.training.v3.grpo_sampler
   :members: run_grpo_sampler_v3

.. automodule:: gpatch.training.v3.grpo_rm
   :members: run_grpo_rm_v3

.. automodule:: gpatch.training.v3.grpo_gen_rm
   :members: run_grpo_gen_rm_v3


.. autoclass:: gpatch.training.v3.ppo_actor.PPOActorTrainerV3
   :members: __init__, hook_after_sampling, hook_before_computing_metrics

.. autoclass:: gpatch.training.v3.grpo_sampler.GrpoSamplerV3
   :members: __init__

.. autoclass:: gpatch.training.v3.grpo_rm.GrpoRmTrainerV3
   :members: __init__

.. autoclass:: gpatch.training.v3.grpo_gen_rm.GrpoGenRmTrainerV3
   :members: __init__

.. autoclass:: gpatch.training.v3.replay_buffer.ReplayBufferWithDataSource
   :members: convert_replay_sample_to_batch_data, convert_partial_sample_to_batch_data
