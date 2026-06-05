Training Backend
================

The training backend module provides engine abstractions for different distributed training frameworks.

Megatron-Core Backend
---------------------

.. autoclass:: gpatch_v4.training_backend.megatron_backend.mcore_engine.McoreEngine
   :members: build_model_and_optimizer, build_train_valid_test_data_iter, train_step, save_ckpt, load_ckpt
   :show-inheritance:

FSDP2 Backend
-------------

.. autoclass:: gpatch_v4.training_backend.fsdp2_backend.fsdp2_engine_lm.Fsdp2EngineLm
   :members:
   :show-inheritance:

.. autoclass:: gpatch_v4.training_backend.fsdp2_backend.fsdp2_engine_t2i.Fsdp2EngineT2i
   :members:
   :show-inheritance:

Loss Factory
------------

.. automodule:: gpatch_v4.training_backend.loss_factory
   :members:

Common Utilities
^^^^^^^^^^^^^^^^

.. autoclass:: gpatch_v4.training_backend.common.swap_mixin.EngineSwapMixin
   :members:
