Generation Backend
==================

Unified inference engine abstraction for rollout generation.

InferEngine
-----------

.. autoclass:: gpatch_v4.generation_backend.infer_engine.InferEngine
   :members: get_engine, build_sampling_params, build_sampling_params_from_config, create_from_engine_args
   :show-inheritance:

SGLang Engine
-------------

.. autoclass:: gpatch_v4.generation_backend.sglang_engine.SglangEngine
   :members: async_generate, get_outputs, flush_cache, resume_memory_occupation, release_memory_occupation, switch_rm_model, update_weights_from_tensor, init_weights_update_group, recv_weights, destroy_weights_update_group, save_checkpoint
   :show-inheritance:

vLLM Engine
-----------

.. autoclass:: gpatch_v4.generation_backend.vllm_engine.VllmEngine
   :members:
   :show-inheritance:

Router Experts Utils
--------------------

.. automodule:: gpatch_v4.generation_backend.routed_experts_utils
   :members:
