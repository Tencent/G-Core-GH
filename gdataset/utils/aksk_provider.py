# sts_provider.py
import json
import os
import time

import requests
import urllib3

# 禁用InsecureRequestWarning警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Global cache for credential provider instances
# Key: (asset_name, access_point, application_name)
_credential_provider_cache = {}


def get_credential_provider(
    asset_name, access_point, application_name, platform_type="gemini", **kwargs
):
    """
    Factory function to get or create a credential provider instance.
    Ensures that the same configuration reuses the same provider instance,
    making the credential cache effective across multiple CosClient instances.

    Args:
        asset_name: Asset name for STS
        access_point: Access point (e.g., "ap-shanghai")
        application_name: Application name
        platform_type: Platform type, either "gemini" or "taiji"
        **kwargs: Additional arguments for specific provider
            For "gemini" platform (acverify provider):
                - No additional parameters needed (wx_ac_ticket from env)
            For "taiji" platform (cert provider):
                - certfile: Certificate file path (required)
                - keyfile: Private key file path (required)

    Returns:
        Credential provider instance with get_credentials() method
    """
    # Create cache key based on asset configuration
    cache_key = (asset_name, access_point, application_name)

    if cache_key not in _credential_provider_cache:
        if platform_type == "gemini":
            _credential_provider_cache[cache_key] = AkskStsCredentialProvider(
                asset_name, access_point, application_name
            )
        elif platform_type == "taiji":
            _credential_provider_cache[cache_key] = CertBasedCredentialProvider(
                asset_name, access_point, application_name, **kwargs
            )
        else:
            raise ValueError(f"Unknown platform_type: {platform_type}. Must be 'gemini' or 'taiji'")

    return _credential_provider_cache[cache_key]


class AkskStsCredentialProvider:
    """aksk temporary credential provider with cache support (ref: https://iwiki.woa.com/p/4016024176)"""
    def __init__(
        self,
        asset_name,
        access_point,
        application_name,
        wx_ac_ticket=None,
        sts_expire_time=60 * 30,
        max_try=60
    ):
        if wx_ac_ticket is None:
            wx_ac_ticket = os.getenv("YARD_APP_SERVICE_TICKET")
        assert wx_ac_ticket is not None, "wx_ac_ticket from env var YARD_APP_SERVICE_TICKET is required"

        self.url = "http://acverify.woa.com:20086/acl/cgi-bin/getststoken?f=json"
        self.headers = {
            "Host": "acverify.woa.com:20086",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Wx-Ac-Ticket": wx_ac_ticket
        }
        self.data = {
            "access_point": access_point,
            "application_name": application_name,
            "asset_name": asset_name,
            "policy":
                {
                    "version":
                        "3.0",
                    "statement":
                        [
                            {
                                "effect": "allow",
                                "action":
                                    [
                                        "name/cos:PutObject", "name/cos:GetObject",
                                        "name/cos:HeadObject", "name/cos:GetBucket",
                                        "name/cos:HeadBucket", "name/cos:DeleteObject"
                                    ],
                                "resource": ["*"]
                            }
                        ]
                }
        }
        self.max_try = max_try
        self.sts_expire_time = sts_expire_time
        self.cache_sts = None

    def _get_sts_realtime(self):
        try:
            response = requests.post(
                self.url, headers=self.headers, data=json.dumps(self.data), timeout=10
            )
            response.raise_for_status()
            return response.json()['sts_info']
        except requests.exceptions.RequestException as e:
            print(f"STS request error: {e}")
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
        return None

    def get_credentials(self):
        """Get credentials with cache, refresh if expired"""
        now_time = int(time.time())

        # Cache miss or about to expire, refresh credentials
        if self.cache_sts is None or \
           self.cache_sts['expired_time'] - now_time < self.sts_expire_time:
            for try_num in range(self.max_try):
                sts_ret = self._get_sts_realtime()
                if sts_ret is not None:
                    self.cache_sts = sts_ret
                    break
                print(f"get sts from acverify.woa.com fail, try_num: {try_num}")
            else:
                raise RuntimeError(f"Failed to get STS credentials after {self.max_try} retries")

        return (
            self.cache_sts['tmp_secret_id'], self.cache_sts['tmp_secret_key'],
            self.cache_sts['token']
        )


class CertBasedCredentialProvider:
    """Certificate-based STS credential provider with cache support"""
    def __init__(
        self,
        asset_name,
        access_point,
        application_name,
        certfile="aigc_client.crt",
        keyfile="aigc_private.key"
    ):
        """
        Initialize certificate-based STS credential provider

        Args:
            asset_name: Asset name for STS
            access_point: Access point (e.g., "ap-shanghai")
            application_name: Application name
            certfile: Certificate file path
            keyfile: Private key file path
        """
        # Hardcoded URL for Taiji STS service
        self.url = "https://shanghai.weixinapi.woa.com:12137/cgi-bin/GetStsToken?appname=wx_aigcdata_sts_token&f=json"

        # Assert certificate files exist
        assert os.path.exists(certfile), f"Certificate file not found: {certfile}"
        assert os.path.exists(keyfile), f"Private key file not found: {keyfile}"

        self.certfile = certfile
        self.keyfile = keyfile
        self.asset_name = asset_name
        self.access_point = access_point
        self.application_name = application_name

        # Use default values
        self.request_timeout = 10
        self.max_try = 60
        self.sts_expire_time = 300

        # Hardcoded default policy
        self.policy = {
            "version":
                "3.0",
            "statement":
                [
                    {
                        "effect": "allow",
                        "action":
                            [
                                "name/cos:PutObject", "name/cos:GetObject", "name/cos:HeadObject",
                                "name/cos:GetBucket", "name/cos:HeadBucket", "name/cos:DeleteObject"
                            ],
                        "resource": ["*"]
                    }
                ]
        }

        self.cache_sts = None

    def _get_sts_realtime(self):
        """Get STS credentials from certificate-based service"""
        try:
            # Check if certificate files exist
            if not os.path.exists(self.certfile):
                print(f"Certificate file not found: {self.certfile}")
                return None
            if not os.path.exists(self.keyfile):
                print(f"Key file not found: {self.keyfile}")
                return None

            # Prepare request data
            data = {
                "policy": self.policy,
                "asset_name": self.asset_name,
                "access_point": self.access_point,
                "application_name": self.application_name
            }

            # Create session and send request
            session = requests.Session()
            response = session.post(
                self.url,
                json=data,
                cert=(self.certfile, self.keyfile),
                verify=True,
                timeout=self.request_timeout,
                headers={"Content-Type": "application/json"}
            )

            response.raise_for_status()
            result = response.json()

            if result.get('ret') == 0:
                return result['sts_info']
            else:
                print(f"STS request failed: {result}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"STS request error: {e}")
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

        return None

    def get_credentials(self):
        """Get credentials with cache, refresh if expired"""
        now_time = int(time.time())

        # Cache miss or about to expire, refresh credentials
        if self.cache_sts is None or \
           self.cache_sts['expired_time'] - now_time < self.sts_expire_time:
            for try_num in range(self.max_try):
                sts_ret = self._get_sts_realtime()
                if sts_ret is not None:
                    self.cache_sts = sts_ret
                    break
                print(f"Get STS from certificate-based service failed, try_num: {try_num}")
            else:
                raise RuntimeError(f"Failed to get STS credentials after {self.max_try} retries")

        return (
            self.cache_sts['tmp_secret_id'], self.cache_sts['tmp_secret_key'],
            self.cache_sts['token']
        )


'''
def test_gemini_provider():
    """Test Gemini platform (aksk) credential provider"""
    print("=" * 80)
    print("Testing Gemini Platform Credential Provider")
    print("=" * 80)

    print("\n[Test 1] Getting Gemini credentials...")
    print("Note: Requires YARD_APP_SERVICE_TICKET environment variable")

    try:
        gemini_provider = get_credential_provider(
            asset_name="s_wxg_aigcdata_xxx",
            access_point="ap-shanghai",
            application_name="p_mmvisionaigcdata",
            platform_type="gemini"
        )

        tmp_secret_id, tmp_secret_key, token = gemini_provider.get_credentials()
        print("✓ Gemini credentials obtained successfully!")
        print(f"  - tmp_secret_id: {tmp_secret_id[:20]}..." if tmp_secret_id else "  - tmp_secret_id: None")
        print(f"  - tmp_secret_key: {tmp_secret_key[:20]}..." if tmp_secret_key else "  - tmp_secret_key: None")
        print(f"  - token: {token[:50]}..." if token else "  - token: None")

        # Test caching
        print("\n[Test 2] Testing credential caching...")
        gemini_provider_2 = get_credential_provider(
            asset_name="s_wxg_aigcdata_xxx",
            access_point="ap-shanghai",
            application_name="p_mmvisionaigcdata",
            platform_type="gemini"
        )

        if gemini_provider is gemini_provider_2:
            print("✓ Provider caching works! Same instance returned.")
        else:
            print("✗ Provider caching failed! Different instance returned.")

        tmp_secret_id_2, tmp_secret_key_2, token_2 = gemini_provider_2.get_credentials()
        print("✓ Cached credentials retrieved successfully!")

    except Exception as e:
        print(f"✗ Gemini credentials failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("Gemini Testing Completed!")
    print("=" * 80)


def test_taiji_provider():
    """Test Taiji platform (cert-based) credential provider"""
    print("=" * 80)
    print("Testing Taiji Platform Credential Provider")
    print("=" * 80)

    print("\n[Test 1] Getting Taiji credentials...")
    print("Note: Requires aigc_client.crt and aigc_private.key files")

    try:
        taiji_provider = get_credential_provider(
            asset_name="s_wxg_aigcdata_zoexqzhou",
            access_point="ap-shanghai",
            application_name="p_mmvisionaigcdata",
            platform_type="taiji",
            certfile="aigc_client.crt",
            keyfile="aigc_private.key"
        )

        tmp_secret_id, tmp_secret_key, token = taiji_provider.get_credentials()
        print("✓ Taiji credentials obtained successfully!")
        print(f"  - tmp_secret_id: {tmp_secret_id[:20]}..." if tmp_secret_id else "  - tmp_secret_id: None")
        print(f"  - tmp_secret_key: {tmp_secret_key[:20]}..." if tmp_secret_key else "  - tmp_secret_key: None")
        print(f"  - token: {token[:50]}..." if token else "  - token: None")

        # Test caching
        print("\n[Test 2] Testing credential caching...")
        taiji_provider_2 = get_credential_provider(
            asset_name="s_wxg_aigcdata_zoexqzhou",
            access_point="ap-shanghai",
            application_name="p_mmvisionaigcdata",
            platform_type="taiji",
            certfile="aigc_client.crt",
            keyfile="aigc_private.key"
        )

        if taiji_provider is taiji_provider_2:
            print("✓ Provider caching works! Same instance returned.")
        else:
            print("✗ Provider caching failed! Different instance returned.")

        tmp_secret_id_2, tmp_secret_key_2, token_2 = taiji_provider_2.get_credentials()
        print("✓ Cached credentials retrieved successfully!")

    except Exception as e:
        print(f"✗ Taiji credentials failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("Taiji Testing Completed!")
    print("=" * 80)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        if test_type == "gemini":
            test_gemini_provider()
        elif test_type == "taiji":
            test_taiji_provider()
        else:
            print(f"Unknown test type: {test_type}")
            print("Usage: python aksk_provider.py [gemini|taiji]")
            print("  gemini - Test Gemini platform (aksk) provider")
            print("  taiji  - Test Taiji platform (cert-based) provider")
            print("  (no args) - Run both tests")
    else:
        # Run both tests if no argument provided
        test_gemini_provider()
        print("\n")
        test_taiji_provider()
'''
