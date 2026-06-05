import json
import uuid
from PIL import Image
from io import BytesIO
import time

import fastapi
import uvicorn
import torch

from megatron.training.initialize import initialize_megatron
from megatron.core import mpu

from gpatch.rpc import once_rpc
from gpatch.patch_mcore import init_gpatch_for_mcore
from megatron.training.global_vars import get_args
from gpatch.training.global_vars import set_global_variables
from gpatch.core.parallel_state import (
    init_pg,
    cpu_barrier,
    is_mp_head,
)
from gpatch.training.utils import (
    print_with_rank_and_datetime,
    find_process_using_port,
)
from gpatch.training.arguments import gpatch_extra_args
from gpatch.core.sampler_v3.infer_engine import InferEngine


def run_eval_server():
    args = get_args()
    ep_ip = args.ppo_sampler_ips[mpu.get_data_parallel_rank()]
    ep_port = args.ppo_sampler_ports[mpu.get_data_parallel_rank()]

    monitor_kwargs = {
        "do_monitor": args.do_monitor,
        "monitor_server_ip": args.monitor_server_ip,
        "monitor_port": args.monitor_port,
    }

    app = fastapi.FastAPI()

    @app.post("/generate")
    @once_rpc(**monitor_kwargs)
    async def generate(req_dict: fastapi.Request) -> fastapi.Response:
        args = get_args()
        use_vllm = args.infer_engine_impl == "vllm"

        sampling_params = infer_engine.get_sampling_params(**req_dict["sampling_params"])

        pillow_imgs = [Image.fromarray(img) for img in req_dict["image"]]
        if use_vllm:
            llm_input = dict(prompt=req_dict["prompt"])
        else:
            llm_input = dict(prompt_token_ids=req_dict["prompt_token_ids"])
        if len(pillow_imgs) > 0:
            llm_input.update({
                "multi_modal_data": {
                    "image": pillow_imgs,
                },
            })

        generation_outputs = [
            infer_engine.async_generate(llm_input, sampling_params, str(uuid.uuid4().hex))
        ]
        gen_outputs = await infer_engine.wait_and_get_async_generate_output(generation_outputs)
        output_token_ids = list(gen_outputs[0].outputs[0].token_ids)

        return dict(output_token_ids=output_token_ids)

    def serve_forever_fn():
        uvicorn.run(
            app,
            host=ep_ip,
            port=ep_port,
            log_level='debug',
            use_colors=False,
            timeout_keep_alive=args.ppo_sampler_server_timeout_keep_alive,
            ssl_keyfile=None,
            ssl_certfile=None,
            ssl_ca_certs=None,
            ssl_cert_reqs=None
        )

    find_process_using_port(ep_ip, ep_port)
    print_with_rank_and_datetime(f'run_grpo_sampler_server http://{ep_ip}:{ep_port}')
    serve_forever_fn()


def eval_sampler_model_provider():
    args = get_args()
    dtype = 'bfloat16'
    tp_size = args.tensor_model_parallel_size
    pp_size = args.pipeline_model_parallel_size
    use_fast = args.px_use_fast_tokenizer
    # https://github.com/vllm-project/vllm/issues/17902
    assert tp_size == args.ppo_sampler_tensor_model_parallel_size
    assert pp_size == args.ppo_sampler_pipeline_model_parallel_size

    return InferEngine.from_engine_args(
        args.infer_engine_impl,
        model=args.load,
        dtype=dtype,
        distributed_executor_backend='ray',
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        gpu_memory_utilization=args.sampler_gpu_memory_utilization,
        enforce_eager=True,
        tp_rank=mpu.get_tensor_model_parallel_rank(),
        pp_rank=mpu.get_pipeline_model_parallel_rank(),
        dp_rank=mpu.get_data_parallel_rank(),
        dist_init_addr=args.sampler_dist_init_addrs[mpu.get_data_parallel_rank()],
        infer_engine_role="sampler",
        use_fast=use_fast,
        max_running_requests=args.max_running_requests,
        load_format="auto",
        allow_auto_truncate=args.sglang_allow_auto_truncate,
        skip_tokenizer_init=not args.sglang_do_tokenizer_init,
        sgl_attention_backend=args.sglang_attention_backend,
        tool_call_parser=args.tool_call_parser,
        enable_return_routed_experts=args.moe_router_replay,
        enable_weights_cpu_backup=True if
        (args.ppo_dynamic_sampling and args.ppo_dynamic_sampling_max_retry > 0) else False,
        sgl_crash_dump_folder=args.sgl_crash_dump_folder,
    )


if __name__ == '__main__':
    init_gpatch_for_mcore()
    initialize_megatron(extra_args_provider=gpatch_extra_args)
    args = get_args()
    args.rl_role = 'sampler'
    set_global_variables(args)
    init_pg(distributed_timeout_minutes=args.distributed_timeout_minutes)

    if args.infer_engine_impl == 'sglang':
        from gpatch.core.sampler_v3.sglang import sglang_hack
        sglang_hack()

    cpu_barrier()
    if is_mp_head():
        infer_engine = eval_sampler_model_provider()
    else:
        if args.infer_engine_impl == 'sglang' and mpu.get_tensor_model_parallel_rank(
        ) % args.num_gpus_per_node == 0:
            infer_engine = eval_sampler_model_provider()

    cpu_barrier()
    print_with_rank_and_datetime(
        f"Init: memory trace {torch.cuda.memory_allocated() / (1024**3)} GB"
    )
    if is_mp_head():
        run_eval_server()
    else:
        while True:
            time.sleep(10)
