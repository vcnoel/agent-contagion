from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REGISTRY = {
    "smol-135m":  "HuggingFaceTB/SmolLM2-135M-Instruct",
    "smol-360m":  "HuggingFaceTB/SmolLM2-360M-Instruct",
    "smol-1.7b":  "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "qwen-0.5b":  "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen-1.5b":  "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen-3b":    "Qwen/Qwen2.5-3B-Instruct",
    "qwen-7b":    "Qwen/Qwen2.5-7B-Instruct",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b":   "Qwen/Qwen3-4B",
    "qwen3.5-0.8b": "Qwen/Qwen3.5-0.8B",
    "qwen3.5-2b":   "Qwen/Qwen3.5-2B",
    "qwen3.5-4b":   "Qwen/Qwen3.5-4B",
    "qwen3.5-9b":   "Qwen/Qwen3.5-9B",
    "llama-1b":   "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3b":   "meta-llama/Llama-3.2-3B-Instruct",
    "llama-8b":   "meta-llama/Llama-3.1-8B-Instruct",
    "gemma-2b":   "google/gemma-2-2b-it",
    "gemma-9b":   "google/gemma-2-9b-it",
    "olmo-1b":    "allenai/OLMo-2-0425-1B-Instruct",
    "olmo-7b":    "allenai/OLMo-2-1124-7B-Instruct",
    "gemma-4-e2b":  "google/gemma-4-E2B-it",
    "smol3-3b":     "HuggingFaceTB/SmolLM3-3B",
}


def load(name: str, dtype=torch.bfloat16):
    # An "@8bit" suffix loads the model bitsandbytes quantised, for the
    # checkpoints that do not fit a 16GB card in bf16. Rows produced this way
    # are reported as quantised in the paper.
    quant = name.endswith("@8bit")
    if quant:
        name = name[:-5]
    repo = REGISTRY.get(name, name)
    tok = AutoTokenizer.from_pretrained(repo, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if quant:
        from transformers import BitsAndBytesConfig
        model = AutoModelForCausalLM.from_pretrained(
            repo, device_map="cuda",
            quantization_config=BitsAndBytesConfig(load_in_8bit=True))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            repo, dtype=dtype, device_map="cuda")
    model.eval()
    return tok, model
