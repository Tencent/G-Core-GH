Extended Model
==============

Extended model wrappers for applying rollout attribute hooks and multi-modal processing.

Base
----

.. autoclass:: gpatch_v4.extended_model.base.ApplySamplingRolloutAttrBase
   :members: apply_rollout_attr, remove_rollout_attr, restore_rollout_attr, restore_all_rollout_attr

LLM Extended Model
-------------------

.. automodule:: gpatch_v4.extended_model.llm
   :members:

Multi-Modal Extended Model
---------------------------

.. automodule:: gpatch_v4.extended_model.multi_modal
   :members:

Qwen3-VL Extended Model
-------------------------

.. automodule:: gpatch_v4.extended_model.qwen3_vl
   :members:

Rollout Attribute Hook
-----------------------

.. automodule:: gpatch_v4.extended_model.rollout_attr_hook
   :members:


Extended Pipeline
=================

Pipeline abstractions for multi-model workflows (text-to-image, text-to-video).

Base Pipeline
-------------

.. autoclass:: gpatch_v4.extended_pipeline.pipeline_base.ExtendedPipelineAbc
   :members: setup, encode_text, repeat_interleave_rollout_batch, permute_timestep_fields
   :show-inheritance:

Flux Pipeline
-------------

.. autoclass:: gpatch_v4.extended_pipeline.pipeline_flux.FluxPipeline
   :members:
   :show-inheritance:

Bagel Pipeline
--------------

.. autoclass:: gpatch_v4.extended_pipeline.pipeline_bagel.BagelPipeline
   :members:
   :show-inheritance:

Pipeline Mixin
--------------

.. automodule:: gpatch_v4.extended_pipeline.mixin
   :members:
