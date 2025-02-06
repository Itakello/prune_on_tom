import os

from transformers import AutoTokenizer

from submodules.SparseLLM.analysis_utils import (
    get_activation_statistics,
    get_attention_heads_statistics,
    get_layerwise_weight_distribution,
    get_sensitivity_analysis,
    run_eval_zero_shot_all,
)


def analyze_models():
    cache_dir = r"D:\tmp\hf_cache"
    # Build full paths for each model directory
    # model_paths = []
    model_paths = [
        os.path.join(cache_dir, d)
        for d in os.listdir(cache_dir)
        if os.path.isdir(os.path.join(cache_dir, d))
    ]
    # Add models inside "D:\tmp\hf_cache\hub"
    model_paths = [p for p in model_paths if "models--" in p]
    model_paths.append("meta-llama/Llama-3.1-8b-Instruct")
    model_paths.append("meta-llama/Llama-3.2-3b-Instruct")

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8b-Instruct")

    # Ensure tokenizer has a pad_token
    if tokenizer.pad_token is None:
        # Option 1: Use the eos_token as the pad token.
        tokenizer.pad_token = tokenizer.eos_token

    get_layerwise_weight_distribution(model_paths)
    get_activation_statistics(model_paths)
    get_sensitivity_analysis(model_paths)
    run_eval_zero_shot_all(model_paths, tokenizer)
    get_attention_heads_statistics(model_paths)


if __name__ == "__main__":
    analyze_models()
