import asyncio

from gpatch_v4 import orches
from gpatch_v4.actor.inference_actor import InferenceWorker
from gpatch_v4.configs.config import InferenceConfig
from gpatch_v4.orches.placement_group import create_infer_group, create_placement_groups


class InferenceRunner:
    async def start(self, config: InferenceConfig):
        print("Step 1: Initializing orches...")
        orches.init()

        print("Step 2: Creating placement groups...")
        pgs = create_placement_groups(config)

        print("Step 3: Creating infer group (sampler)...")
        infer_group = create_infer_group(config, pgs)

        print("Step 4: Serving...")
        await infer_group.init()

        #await asyncio.Event().wait()

        infer_worker = InferenceWorker()
        print("Step 5: Initializing inference actor group...")
        await infer_worker.init(config)

        print("Step 6: Setting up sampler client...")
        await infer_worker.setup_client()

        print("Step 7: Running inference...")
        await infer_worker.inference()

        print("Inference completed!")
