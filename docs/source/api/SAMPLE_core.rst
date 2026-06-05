Core Module (``gpatch_v4.core``)
=================================

This module provides core training algorithms and utilities, including advantage computation,
device management, and extensibility registries.

Advantage Computation
---------------------

Context and Results
^^^^^^^^^^^^^^^^^^^

.. autoclass:: gpatch_v4.core.advantage_helper.AdvantageContext
   :members:
   :undoc-members:

.. autoclass:: gpatch_v4.core.advantage_helper.AdvantageResult
   :members:
   :undoc-members:

Functions
^^^^^^^^^

.. automodule:: gpatch_v4.core.advantage_helper
   :members: register_advantage, get_advantage_fn, register_custom_advantage
   :undoc-members:

.. automodule:: gpatch_v4.core.advantage_impl
   :members: calculate_grpo_advantages, calculate_ppo_rewards, calculate_ppo_advantages_and_returns, calculate_reverse_kl_advantages, calculate_g_opd_advantages
   :undoc-members:

Device Management
-----------------

.. automodule:: gpatch_v4.core.device
   :members:
   :undoc-members:

.. automodule:: gpatch_v4.core.parallel_state
   :members:
   :undoc-members:

Training Utilities
------------------

.. automodule:: gpatch_v4.core.correction_helper
   :members: compute_off_policy_correction_weights
   :undoc-members:

.. automodule:: gpatch_v4.core.smart_pad_helper
   :members:
   :undoc-members:

.. automodule:: gpatch_v4.core.seqlen_balancing
   :members:
   :undoc-members:

Constants
---------

.. autoclass:: gpatch_v4.core.constants.MODEL_ARCH
   :members:
   :undoc-members:

Built-in Advantage Types
-------------------------

.. automodule:: gpatch_v4.core
   :members: BUILDIN_ADVANTAGE_TYPE
   :noindex:
