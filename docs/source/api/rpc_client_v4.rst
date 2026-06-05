RPC Client
==========

RPC client abstractions for inter-component communication.

Base Classes
------------

.. autoclass:: gpatch_v4.rpc_client.base_rpc_client.RpcClient
   :members: execute, get_target_endpoint, pick_endpoint_index
   :show-inheritance:

.. autoclass:: gpatch_v4.rpc_client.base_rpc_client.HttpRpcClient
   :members: execute, build_url
   :show-inheritance:

.. autoclass:: gpatch_v4.rpc_client.base_rpc_client.ZeroMqRpcClient
   :members:
   :show-inheritance:

Ray RPC Client
--------------

.. autoclass:: gpatch_v4.rpc_client.ray_rpc_client.RayRpcClient
   :members:
   :show-inheritance:

Multi-Cast Ray RPC Client
--------------------------

.. autoclass:: gpatch_v4.rpc_client.multi_cast_ray_rpc_client.MultiCastRayRpcClient
   :members:
   :show-inheritance:
