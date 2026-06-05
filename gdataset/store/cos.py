# copyright (c) 2024 tencent inc. all rights reserved.
# nrwu@tencent.com

import logging
import os
import time
import traceback
from datetime import datetime
from typing import Dict

import requests
import urllib3
from typing_extensions import override

try:
    import qcloud_cos
    from qcloud_cos import CosConfig, CosS3Client
    from qcloud_cos.cos_client import logger
    from qcloud_cos.cos_exception import CosServiceError
    logger.setLevel(logging.WARNING)
except ImportError:
    pass

# STS credential error codes that require credential refresh
STS_CREDENTIAL_ERROR_CODES = {'InvalidAccessKeyId', 'KmsKeyDisabled', 'KmsKeyNotExist'}
try:
    from gdataset.utils.service_discovery import ServiceInstanceCache
except ImportError:
    ServiceInstanceCache = None

from gdataset.store.base import CliBase


class CosClient(CliBase):
    def __init__(self, metadata, **kwargs):
        self.cos_region = metadata.get('cos_region', None)
        self.cos_aksk_provider = None  # Initialize cos_aksk_provider
        credential_type = metadata.get('cos_credential_type', 'static')  # static or aksk
        if credential_type == 'aksk':  # aksk方式，即临时加密密钥
            # Dynamic credential types that require STS provider
            from gdataset.utils.aksk_provider import get_credential_provider

            # Get config using one pattern: cos_aksk_config
            config = metadata['cos_aksk_config']

            # get platform_type with env
            if os.getenv("YARD_APP_SERVICE_TICKET") is not None:
                platform_type = 'gemini'
                provider_kwargs = {}
            else:
                platform_type = 'taiji'
                provider_kwargs = {
                    'certfile': config['certfile'],
                    'keyfile': config['keyfile'],
                }

            self.cos_aksk_provider = get_credential_provider(
                asset_name=config['asset_name'],
                access_point=config['access_point'],
                application_name=config['application_name'],
                platform_type=platform_type,
                **provider_kwargs
            )
            self.cos_secret_id, self.cos_secret_key, self.token = self.cos_aksk_provider.get_credentials(
            )

        elif credential_type == 'static':  # static方式，即明文密钥
            # Static credential type
            self.cos_secret_id = metadata['cos_secret_id']
            self.cos_secret_key = metadata['cos_secret_key']
            self.token = metadata.get('cos_token', None)
        else:
            raise ValueError(f'Unsupported credential type: {credential_type}')

        self.service_domain = metadata.get("service_domain", None)
        self.cos_endpoint = metadata.get('cos_endpoint', None)
        self.cos_service = metadata.get('cos_service', None)
        self.svr_discovery = None

        # 根据 cos client 代码，cos client 内部会维护进程级别的连接池，而 client 的 ctor 是非常
        # 轻量级的。
        self.cos_clis: Dict[str, CosS3Client] = {}
        if self.cos_region is not None:
            self.cos_clis[self.cos_region] = self.create_client(self.cos_region)

    def _refresh_credentials_and_clients(self):
        """Refresh aksk credentials and rebuild all COS clients"""
        if self.cos_aksk_provider is None:
            return False

        # Force refresh credentials from STS provider
        self.cos_aksk_provider.cache_sts = None
        self.cos_secret_id, self.cos_secret_key, self.token = self.cos_aksk_provider.get_credentials(
        )
        self.cos_clis[self.cos_region] = self.create_client(self.cos_region)
        return True

    def create_client(self, cos_region) -> CosS3Client:
        scheme = 'https' if self.cos_service is None else 'http'
        cos_endpoint = self.cos_endpoint
        if self.cos_endpoint is None:
            # 腾讯机房内网
            cos_endpoint = 'cos-internal.%s.tencentcos.cn' % cos_region
        cos_cli_config = CosConfig(
            Region=cos_region,
            SecretId=self.cos_secret_id,
            SecretKey=self.cos_secret_key,
            Token=self.token,
            Endpoint=cos_endpoint,
            Scheme=scheme,
            ServiceDomain=self.service_domain,
            Timeout=30,
        )
        return CosS3Client(cos_cli_config)

    def _get(self, cos_url='', url='', cos_bucket_name='', **kwargs):
        if cos_url != '':
            assert url == ''
            url = cos_url

        if self.cos_service is not None and self.svr_discovery is None:
            assert ServiceInstanceCache is not None
            self.svr_discovery = ServiceInstanceCache()

        cos_region = kwargs.get("cos_region", self.cos_region)
        if cos_region not in self.cos_clis and cos_region is not None:
            self.cos_clis[cos_region] = self.create_client(cos_region)
        cos_client = self.cos_clis[cos_region]
        if self.cos_service is not None:
            cos_instance = self.svr_discovery.get_instance("Production", self.cos_service)
            cos_client.get_conf().set_ip_port(cos_instance['host'], cos_instance['port'])

        cos_resp = cos_client.get_object(
            Bucket=cos_bucket_name,
            Key=url,
        )
        rt = cos_resp['Body']._rt
        start = time.time()
        total_read_timeout = 60
        chunks = []
        try:
            for chunk in rt.iter_content(chunk_size=65536):
                if chunk:
                    chunks.append(chunk)
                if time.time() - start > total_read_timeout:
                    raise TimeoutError(f"COS body read exceeded {total_read_timeout}s for {url}")
        finally:
            rt.close()
        return b''.join(chunks)

    def get(self, cos_url='', url='', cos_bucket_name='', **kwargs):
        perf = int(os.environ.get("GDATASET_V4_PERF", "0"))
        if perf == 1:
            begin_t = time.time()

        base_delay = 0.016  # 16ms initial delay
        max_retries = 10
        credential_refreshed = False  # Track if we've already refreshed credentials in this request

        for attempt in range(max_retries):
            try:
                data = self._get(
                    cos_url=cos_url, url=url, cos_bucket_name=cos_bucket_name, **kwargs
                )

                if perf == 1:
                    end_t = time.time()
                    during_t = end_t - begin_t
                    if during_t > 0.1:
                        print(f"cos get {url} cost {during_t:.5f} seconds")
                return data

            except CosServiceError as e:
                error_code = e.get_error_code()
                time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # Check if it's a credential-related error that requires refresh
                if error_code in STS_CREDENTIAL_ERROR_CODES and not credential_refreshed:
                    print(
                        f'Warning(cos) {time_str=} CosServiceError with credential error: {error_code}, '
                        f'attempting to refresh credentials. input: {cos_url=} {url=} {cos_bucket_name=}',
                        flush=True,
                    )
                    if self._refresh_credentials_and_clients():
                        credential_refreshed = True
                        continue  # Retry immediately with new credentials

                # For other CosServiceError or if refresh already attempted
                print(
                    f'Warning(cos) {time_str=} CosServiceError: {error_code} - {e.get_error_msg()} '
                    f'retry {attempt} input: {cos_url=} {url=} {cos_bucket_name=} {kwargs=}',
                    flush=True,
                )
                if attempt >= max_retries - 1:
                    break
                time.sleep(self._calc_backoff(attempt, base_delay))

            except (
                qcloud_cos.cos_exception.CosClientError, urllib3.exceptions.ReadTimeoutError,
                requests.exceptions.ConnectionError, TimeoutError, Exception
            ) as e:
                time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(
                    f'Warning(cos) {time_str=} cos get error: {type(e).__name__}: {e} '
                    f'retry {attempt}/{max_retries} input: {cos_url=} {url=} {cos_bucket_name=} {kwargs=}',
                    flush=True,
                )
                if attempt >= max_retries - 1:
                    break
                time.sleep(self._calc_backoff(attempt, base_delay))

        raise RuntimeError(f'Failed to get {url} after {max_retries} retries')

    def _calc_backoff(self, attempt, base_delay):
        """Exponential backoff algorithm"""
        return min(
            base_delay * (2**attempt), 16
        )  # Maximum interval 16 seconds (total retry about 15min)
