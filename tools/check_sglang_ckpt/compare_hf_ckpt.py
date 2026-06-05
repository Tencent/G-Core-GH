import argparse
import torch
from transformers import AutoModel, AutoConfig


def load_model_state_dict(model_path: str):
    """Load model with transformers and return its state_dict on CPU."""
    print(f"[INFO] Loading model from: {model_path}")
    # You can switch to AutoModelForCausalLM if needed
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModel.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float32,  # force float32 for deterministic comparison
        device_map=None,  # load on CPU
    )
    model.eval()
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    # Free model to save memory
    del model
    return state_dict


def compare_state_dicts(sd1: dict, sd2: dict) -> bool:
    """Compare two state_dicts. Return True if all tensors are exactly equal."""
    keys1 = set(sd1.keys())
    keys2 = set(sd2.keys())

    ok = True

    if keys1 != keys2:
        only_in_1 = keys1 - keys2
        only_in_2 = keys2 - keys1
        if only_in_1:
            print("[DIFF] Keys only in model1 (first 20):")
            for k in list(sorted(only_in_1))[:20]:
                print("  ", k)
        if only_in_2:
            print("[DIFF] Keys only in model2 (first 20):")
            for k in list(sorted(only_in_2))[:20]:
                print("  ", k)
        ok = False

    # Only compare intersection keys
    common_keys = sorted(keys1 & keys2)
    print(f"[INFO] Number of common tensors: {len(common_keys)}")

    for name in common_keys:
        t1 = sd1[name]
        t2 = sd2[name]

        if t1.shape != t2.shape:
            print(f"[DIFF] Shape mismatch for '{name}': {t1.shape} vs {t2.shape}")
            ok = False
            continue

        if not torch.equal(t1, t2):
            # If you want tolerance, replace with torch.allclose(...)
            max_diff = (t1 - t2).abs().max().item()
            print(f"[DIFF] Value mismatch for '{name}', max abs diff: {max_diff}")
            ok = False

    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Compare weights of two HuggingFace transformers models."
    )
    parser.add_argument("--model_path_1", type=str, help="Path to first model (ckpt dir)")
    parser.add_argument("--model_path_2", type=str, help="Path to second model (ckpt dir)")
    args = parser.parse_args()

    # Load state dicts
    sd1 = load_model_state_dict(args.model_path_1)
    sd2 = load_model_state_dict(args.model_path_2)

    # Compare
    same = compare_state_dicts(sd1, sd2)
    if same:
        print("\n[RESULT] All weights are exactly identical.")
    else:
        print("\n[RESULT] Models are NOT identical.")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
