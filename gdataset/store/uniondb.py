# copyright (c) 2024 tencent inc. all rights reserved.
# nrwu@tencent.com
import os
import time
import traceback
from datetime import datetime

import httpx
from typing_extensions import override

try:
    from gdataset.utils.service_discovery import ServiceInstanceCache
except ImportError:
    ServiceInstanceCache = None

from gdataset.store.base import CliBase


class UnionDBError(RuntimeError):
    """Custom UnionDB exception base class"""
    def __init__(self, message, code=None, retriable=False):
        super().__init__(message)
        self.code = code  # Original error code
        self.retriable = retriable  # Retry flag


def udb_default_hash(x: bytes, seed: int = 271, sum_val: int = 1) -> int:
    """
    Calculate the default hash value for UnionDB partitioning.
    
    Args:
        x: Input bytes for hash calculation
        seed: Hash seed value (default: 271)
        sum_val: Initial value for hash calculation (default: 1)
        
    Returns:
        Computed hash value as 32-bit unsigned integer
    """
    for byte in x:
        sum_val = (sum_val * seed + byte) & 0xFFFFFFFF
    return sum_val


class UniondbClient(CliBase):
    """HTTP client for interacting with UnionDB service."""
    def __init__(self, metadata, **kwargs):
        """
        Initialize UnionDB client with service metadata.
        
        Args:
            metadata: Configuration dictionary containing:
                - udb_endpoint: Base URL of UnionDB service
                - udb_table: Name of the target data table
                - udb_token_key: Authentication token for DB access
        """
        if "udb_endpoint" in metadata:
            self.endpoint = f"{metadata['udb_endpoint']}/Get"
        else:
            self.endpoint = None
            self.udb_namespace = metadata['udb_namespace']
            self.udb_service = metadata['udb_service']
            self.svr_discovery = None

        self.headers = {
            "Content-Type": "application/json",
            "Svrkit_Req_Base64": "0",
            "Svrkit_Resp_Base64": "0"
        }
        self.udb_table = metadata["udb_table"]
        self.udb_token_key = metadata["udb_token_key"]

    def _get(self, **kwargs):
        """
        Retrieve data from UnionDB with retry mechanism.
        
        Args:
            primary_key: Unique identifier for the data record
            columns: List of column names to retrieve (format: "CF_qualifier")
            
        Returns:
            Dictionary of retrieved values keyed by qualifier
            
        Raises:
            RuntimeError: After exhausting all retry attempts
        """
        # Build base request structure
        request_body = {
            "table_name": self.udb_table,
            "reqs": [],
            "param": {
                "token": self.udb_token_key
            }
        }

        # Calculate partition ID
        pkey = kwargs['primary_key'].encode('utf-8')
        part_id = udb_default_hash(pkey, seed=47, sum_val=0) % 32000

        # Build column requests
        cols = kwargs["columns"]

        for column in cols:
            cf, qual = column.split('_', 1)
            request_body["reqs"].append(
                {
                    "part_id": part_id,
                    "sort_key": kwargs['primary_key'],
                    "family": cf,
                    "qualifier": qual,
                    "timestamp": (1 << 64) - 1
                }
            )

        # Retry configuration
        base_delay = 0.016  # 16ms initial delay
        max_retries = 60
        last_exception = None

        if self.endpoint is None and self.svr_discovery is None:
            assert ServiceInstanceCache is not None
            self.svr_discovery = ServiceInstanceCache()

        # Retry loop
        for attempt in range(max_retries):
            try:
                if self.endpoint is None:
                    udb_instance = self.svr_discovery.get_instance(
                        self.udb_namespace, self.udb_service
                    )
                    udb_url = f"http://{udb_instance['host']}:{udb_instance['port']}/Get"
                else:
                    udb_url = self.endpoint
                with httpx.Client(timeout=10) as client:
                    response = client.post(url=udb_url, headers=self.headers, json=request_body)
                    response.raise_for_status()

                    svr_result = response.headers.get("X-SvrKit-Result")
                    if svr_result is not None:
                        try:
                            svr_code = int(svr_result)
                            if svr_code != 0:
                                # Determine error type and generate exception
                                is_retriable = self._is_retriable_error(svr_code)
                                raise UnionDBError(
                                    f"SVRKit RPC error (code={svr_code}) {pkey=}",
                                    code=svr_code,
                                    retriable=is_retriable
                                )
                        except ValueError:
                            raise UnionDBError(
                                f"Invalid X-SvrKit-Result format: {svr_result} {pkey=}",
                                retriable=False
                            )

                    data = response.json()
                    ret_codes = data.get("ret_codes")

                if ret_codes is None:
                    raise UnionDBError(
                        f"Response missing 'ret_codes' field {pkey=}", retriable=False
                    )

                # Check uniondb logic code
                if any(code != 0 for code in ret_codes):
                    error_codes = [c for c in ret_codes if c != 0]
                    # Determine retriable status (adjust based on business requirements)
                    is_retriable = self._is_retriable_error(error_codes)
                    raise UnionDBError(
                        f"Business logic error: {error_codes} {pkey=}",
                        code=error_codes,
                        retriable=is_retriable
                    )
                return self._parse_response(data, cols)

            except UnionDBError as e:
                # Handle custom exceptions
                if not e.retriable:
                    raise  # Directly raise non-retriable errors
                time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                last_exception = e
                print(f'Warning(uniondb) {time_str=} uniondb error {e} retry {attempt}', flush=True)
                # traceback.print_exc()
                if attempt >= max_retries - 1:
                    break
                time.sleep(self._calc_backoff(attempt, base_delay))

            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                # Convert network errors to retriable exceptions
                last_exception = UnionDBError(
                    f"Network communication error: {str(e)} {pkey=}",
                    code=getattr(e, 'status_code', None),
                    retriable=True
                )
                time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f'Warning(uniondb) {time_str=} http error {e} retry {attempt}', flush=True)
                # traceback.print_exc()
                if attempt >= max_retries - 1:
                    break
                time.sleep(self._calc_backoff(attempt, base_delay))

        # Wrap final exception after retries exhausted
        raise UnionDBError(
            f"Request failed after maximum retries ({max_retries}) {pkey=}",
            code=getattr(last_exception, 'code', None),
            retriable=False
        ) from last_exception

    def _is_retriable_error(self, codes) -> bool:
        RETRIABLE_RANGES = [
            (-1000, -200),  # svrkit code
            (-50020, -50000),  # udb proxy code
            (-30030, -30000)  # udb tablet code
        ]

        check_codes = [codes] if isinstance(codes, int) else codes
        return any(
            any(low <= code <= high for (low, high) in RETRIABLE_RANGES)
            for code in check_codes if isinstance(code, int)
        )

    def _calc_backoff(self, attempt, base_delay):
        """Exponential backoff algorithm"""
        return min(
            base_delay * (2**attempt), 16
        )  # Maximum interval 16 seconds (total retry about 15min)

    def _parse_response(self, data, columns):
        """
        Parse UnionDB response into qualifier-keyed dictionary.
        
        Args:
            data: Raw response data from UnionDB
            columns: Original requested columns list
            
        Returns:
            Dictionary mapping qualifiers to their values
        """
        result = {}
        cf_qualifier = [s.split('_', 1) for s in columns]

        for (_, qual), value in zip(cf_qualifier, data["ret_values"]):
            # Handle potential binary responses
            result[qual] = value if isinstance(value, str) else value.decode('utf-8')

        return result

    @override
    def get(self, **kwargs):
        """
        Retrieve data from UnionDB with retry mechanism.
        
        Args:
            primary_key: Unique identifier for the data record
            columns: List of column names to retrieve (format: "CF_qualifier")
            
        Returns:
            Dictionary of retrieved values keyed by qualifier
            
        Raises:
            RuntimeError: After exhausting all retry attempts
        """
        perf = int(os.environ.get("GDATASET_V4_PERF", "0"))
        if perf == 1:
            begin_t = time.time()
        data = self._get(**kwargs)
        if perf == 1:
            end_t = time.time()
            during_t = end_t - begin_t
            if during_t > 0.1:
                print(f"uniondb get cost: {during_t:.5f} seconds")
        return data
