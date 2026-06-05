Actors
======

Ray actor implementations for each training role (train, sampler, reward model, etc.).

Base & Mixins
-------------

.. autoclass:: gpatch_v4.actor.mixin.TokenizerMixin
   :members:

.. autoclass:: gpatch_v4.actor.mixin.MetricsMixin
   :members:

.. autoclass:: gpatch_v4.actor.mixin.CheckpointConverterMixin
   :members:

.. autoclass:: gpatch_v4.actor.mixin.RlTrainerMixin
   :members:

.. autoclass:: gpatch_v4.actor.mixin.RetryActorMixin
   :members:

.. autoclass:: gpatch_v4.actor.mixin.ProfileMixin
   :members:

.. autoclass:: gpatch_v4.actor.mixin.FlopsCounterMixin
   :members:

GRPO Actors
-----------

.. autoclass:: gpatch_v4.actor.grpo_train_actor.GrpoTrainActor
   :members:
   :show-inheritance:

.. autoclass:: gpatch_v4.actor.grpo_async_train_actor.GrpoAsyncTrainActor
   :members:
   :show-inheritance:

.. autoclass:: gpatch_v4.actor.grpo_sampler_actor.GrpoSamplerActor
   :members:
   :show-inheritance:

.. autoclass:: gpatch_v4.actor.grpo_bt_rm_actor.GrpoBtRmActor
   :members:
   :show-inheritance:

.. autoclass:: gpatch_v4.actor.grpo_gen_rm_actor.GrpoGenRmActor
   :members:
   :show-inheritance:

Distillation Actors
-------------------

.. autoclass:: gpatch_v4.actor.distill_student_actor.DistillStudentActor
   :members:
   :show-inheritance:

.. autoclass:: gpatch_v4.actor.distill_teacher_actor.DistillTeacherActor
   :members:
   :show-inheritance:

.. autoclass:: gpatch_v4.actor.off_policy_distill_student_actor.OffPolicyDistillStudentActor
   :members:
   :show-inheritance:

.. autoclass:: gpatch_v4.actor.off_policy_distill_sampler_actor.OffPolicyDistillSamplerActor
   :members:
   :show-inheritance:

Fine-tuning & DPO Actors
-------------------------

.. autoclass:: gpatch_v4.actor.finetune_actor.FinetuneActor
   :members:
   :show-inheritance:

.. autoclass:: gpatch_v4.actor.dpo_actor.DpoActor
   :members:
   :show-inheritance:

Agentic RL Actors
-----------------

.. autoclass:: gpatch_v4.actor.grpo_agentic_train_actor.GrpoAgenticTrainActor
   :members:
   :show-inheritance:

T2I Actors
----------

.. autoclass:: gpatch_v4.actor.t2i_grpo_train_actor.T2iGrpoTrainActor
   :members:
   :show-inheritance:

.. autoclass:: gpatch_v4.actor.t2i_grpo_bt_rm_actor.T2iGrpoBtRmActor
   :members:
   :show-inheritance:

.. autoclass:: gpatch_v4.actor.t2i_grpo_gen_rm_actor.T2iGrpoGenRmActor
   :members:
   :show-inheritance:

Inference & Evaluation
----------------------

.. autoclass:: gpatch_v4.actor.inference_actor.InferenceWorker
   :members:
   :show-inheritance:

.. autoclass:: gpatch_v4.actor.evaluate_actor.EvaluateActor
   :members:
   :show-inheritance:
