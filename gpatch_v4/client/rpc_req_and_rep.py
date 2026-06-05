from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class RequestFormat:
    """ request data structure """
    rpc_type: str
    request_id: str = None
    timestamp: float = None
    request_data: Dict[Any] = None


@dataclass
class ResponseFormat:
    rpc_type: str
    request_id: str = None
    timestamp: float = None
    response_data: Dict[Any] = None
