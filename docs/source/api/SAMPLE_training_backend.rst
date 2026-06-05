Training Backend Module (``gpatch_v4.training_backend``)
=========================================================

Training engine abstractions and implementations for different backends (FSDP2, Megatron-Core).

Training Engine Factory
-----------------------

.. autoclass:: gpatch_v4.training_backend.TrainingEngineFactory
   :members: get_training_engine
   :undoc-members:

Base Engine
-----------

.. autoclass:: gpatch_v4.training_backend.base_engine.BaseEngine
   :members: setup_model_and_get_optimizer, compute_log_probs, rl_train_actor, finetune_step, set_model_eval, set_model_train
   :undoc-members:

FSDP2 Backend
-------------

.. autoclass:: gpatch_v4.training_backend.fsdp2_backend.Fsdp2EngineLm
   :members:
   :undoc-members:

.. autoclass:: gpatch_v4.training_backend.fsdp2_backend.Fsdp2EngineT2i
   :members:
   :undoc-members:

Megatron-Core Backend
---------------------

.. autoclass:: gpatch_v4.training_backend.megatron_backend.McoreEngine
   :members:
   :undoc-members:

Loss Functions
--------------

.. automodule:: gpatch_v4.training_backend.loss_factory
   :members: register_loss, register_custom_loss_fn, opd_loss_func, gspo_loss_func, fipo_loss_func, cross_entroy_loss_func, dpo_loss_func, square_averaging_cross_entroy_loss_func
   :undoc-members:

Built-in Loss Functions
-----------------------

.. automodule:: gpatch_v4.training_backend
   :members: BUILDIN_LOSS_FUNC
   :noindex:
