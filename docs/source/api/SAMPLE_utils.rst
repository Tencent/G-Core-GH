Utils Module (``gpatch_v4.utils``)
===================================

Utility functions and classes for logging, communication, data manipulation, and performance profiling.

Logging and Output
------------------

.. automodule:: gpatch_v4.utils.common_utils
   :members: logging_rank0, logging_with_rank_and_datetime, logging_memory_usage, log
   :undoc-members:

Performance Profiling
---------------------

.. automodule:: gpatch_v4.utils.common_utils
   :members: perf_time, profile_memory_and_time, clear_memory, logging_meminfo_str
   :undoc-members:

Distributed Communication
--------------------------

.. autoclass:: gpatch_v4.utils.communication_utils.BroadcastUtils
   :members:
   :undoc-members:

.. automodule:: gpatch_v4.utils.communication_utils
   :members: all_reduce_autograd, allreduce_loss_across_data_parallel_group, average_losses_across_data_parallel_group, broadcast_2d_tensor
   :undoc-members:

Data Manipulation
-----------------

.. automodule:: gpatch_v4.utils.data_manipulate_utils
   :members: aggregate_metrics, pad_to_length
   :undoc-members:

Dynamic Imports
---------------

.. automodule:: gpatch_v4.utils.common_utils
   :members: import_mod_from_path, import_fn_from_path, safe_import_class
   :undoc-members:

Custom Filtering
----------------

.. autoclass:: gpatch_v4.utils.filter_samplings.FilterSamplingRegistry
   :members:
   :undoc-members:

.. automodule:: gpatch_v4.utils.filter_samplings
   :members: register_custom_filter_sampling, keep_all, truncated_test, best_and_worst
   :undoc-members:

Timing and Reporting
---------------------

.. autoclass:: gpatch_v4.utils.timer_utils.TimerSingleton
   :members:
   :undoc-members:

.. autoclass:: gpatch_v4.utils.report_utils.TrainReporterSingleton
   :members:
   :undoc-members:

Data Envelope
-------------

.. autoclass:: gpatch_v4.utils.common_utils.Envelope
   :members:
   :undoc-members:

Built-in Filter Sampling Strategies
------------------------------------

.. automodule:: gpatch_v4.utils
   :members: BUILDIN_FILTER_SAMPLING_STRATEGIES
   :noindex:
