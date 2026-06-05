import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = str(sys.argv[1])

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
)

print(f"eos_token {tokenizer.eos_token} {tokenizer.eos_token_id}")
messages = [
    {
        "role": "user",
        "content": "Hello, how are you?"
    }, {
        "role": "assistant",
        "content": "I'm doing well, thank you!"
    }
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
print(f"format text {text}")

print("inspect model")
for pname, params in model.named_parameters():
    print(
        f"Trace export_weights {pname=} shape {params.shape=} dtype {params.dtype=} {params.sum()}"
    )
