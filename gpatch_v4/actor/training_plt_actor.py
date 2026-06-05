from gpatch_v4.orches import get_queue
from gpatch_v4.utils import log


class TrainingPltActor:
    def __init__(self):
        # 使用 Ray 提供的分布式 Queue
        self.queue = get_queue(maxsize=1000)
        self._running = True

    def add_data(self, data):
        self.queue.put(data)

    def stop(self):
        self._running = False
        self.queue.put("__STOP__")

    async def run_loop(self):
        log("loop started.")

        while self._running:
            try:
                # 使用 Ray Queue 的异步获取方法
                data = await self.queue.get_async()
                if isinstance(data, str) and data == "__STOP__":
                    log("received stop signal.")
                    break

                # log(f"out: {data=}")

            except Exception as e:
                log(f"Error: {e}")

        log("loop stopped.")
        return "loop_finished"
