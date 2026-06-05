import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

model_paths = [
    "deepseek-v2-lite-hf_release",
    "/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/hf-hub/deepseek-ai/DeepSeek-V2-Lite",
]

# `max_memory` should be set based on your devices
max_memory = {i: "75GB" for i in range(8)}
# `device_map` cannot be set to `auto`
tokenizer = AutoTokenizer.from_pretrained(model_paths[0], trust_remote_code=True)

models = []
for model_path in model_paths:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="sequential",
        torch_dtype=torch.bfloat16,
        max_memory=max_memory,
        attn_implementation="eager"
    )
    model.generation_config = GenerationConfig.from_pretrained(model_path)
    print(f"{model.generation_config=}")
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    models.append(model)

assert len(models) == 2
# print(f"{models[0]=}, {models[1]=} {id(models[0])=}, {id(models[1])=}")

model0 = models[0].state_dict()
model1 = models[1].state_dict()
# print(f"{id(model0)=} {id(model1)=}")

for k in model0.keys():
    v1 = model0[k]
    v2 = model1[k]

    # print(f"[DEBUG] {k=} {v1=} {v2=}")
    # assert False, "debug only"
    assert torch.allclose(v1, v2), print(f"{k} is different {v1=}, {v2=}")

messages = [{"role": "user", "content": "Write a piece of quicksort code in C++"}]
input_tensor = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
)

for model_path, model in zip(model_paths, models):
    outputs = model.generate(input_tensor.to(model.device), max_new_tokens=100, do_sample=False)
    result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
    print(f"{model_path=}\n{outputs=}\n{result=}")
