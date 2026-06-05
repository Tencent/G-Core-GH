import random
import time

# pip install polaris-cpp-py --index-url https://mirrors.cloud.tencent.com/pypi/simple/
from polaris.api.consumer import create_consumer_by_default_config_file
from polaris.pkg.model.error import SDKError
from polaris.pkg.model.service import GetInstancesRequest

# consumer api 内部有缓存，全局只使用一个consumer api对象
g_consumer_api = None


def get_consumer_api():
    global g_consumer_api
    if not g_consumer_api:
        g_consumer_api = create_consumer_by_default_config_file()
    return g_consumer_api


class ServiceInstanceCache:
    def __init__(self):
        self._cache = {}  # {(namespace, service): [instances, index, timestamp]}
        self.consumer_api = get_consumer_api()
        self.update_interval = random.randint(100, 150)

    def get_instance(self, namespace, service):
        key = (namespace, service)
        now = time.time()

        # 初始化或获取缓存条目
        entry = self._cache.get(key)
        if not entry:
            entry = self._init_entry(key, now)

        # 检查是否需要更新（超过120秒）
        if now - entry[2] > self.update_interval:
            self._update_entry(key, now)
            entry = self._cache[key]  # 获取更新后的条目

        # 处理空实例列表
        if not entry[0]:
            raise Exception("No available instances")

        # 轮询逻辑
        current_idx = entry[1]
        selected = entry[0][current_idx]

        # 更新索引（单线程无需锁）
        new_idx = (current_idx + 1) % len(entry[0])
        self._cache[key][1] = new_idx

        return {"host": selected["host"], "port": selected["port"]}

    def _init_entry(self, key, timestamp):
        namespace, service = key
        try:
            instances = self._fetch_instances(namespace, service)
            random.shuffle(instances)
            self._cache[key] = [instances, 0, timestamp]
        except Exception as e:
            print(f"Initialization failed: {e}")
            self._cache[key] = [[], 0, timestamp]
        return self._cache[key]

    def _update_entry(self, key, timestamp):
        namespace, service = key
        try:
            new_instances = self._fetch_instances(namespace, service)
            if not new_instances:
                return

            random.shuffle(new_instances)
            # 直接更新整个条目
            self._cache[key] = [new_instances, 0, timestamp]
            print(f"Updated {namespace}/{service} with {len(new_instances)} instances")
        except Exception as e:
            print(f"Update failed: {e}")

    def _fetch_instances(self, namespace, service):
        """调用Polaris API获取实例"""
        request = GetInstancesRequest(namespace=namespace, service=service)
        try:
            response = self.consumer_api.get_all_instances(request)
            return [
                {
                    "host": instance.get_host(),
                    "port": instance.get_port()
                } for instance in response
            ]
        except SDKError as e:
            raise Exception(f"Polaris SDK error: {repr(e)}")

    def manual_refresh(self, namespace, service):
        """手动刷新接口"""
        self._update_entry((namespace, service), time.time())
