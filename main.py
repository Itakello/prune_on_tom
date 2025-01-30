import argparse
import csv
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from submodules.SparseLLM.datautils import SYSTEM_PROMPT, _build_user_message, get_tom
from submodules.SparseLLM.model_utils import llama_sparsellm

SHORT_NAMES = {
    "Ambiguous Story Task.jsonl": "AST",
    "False Belief Task.jsonl": "FBT",
    "Hinting Task Test.jsonl": "HTT",
    "Faux-pas Recognition Test.jsonl": "FPR",
    "Unexpected Outcome Test.jsonl": "UOT",
    "Persuasion Story Task.jsonl": "PST",
    "Strange Story Task.jsonl": "SST",
    "Scalar Implicature Test.jsonl": "SIT",
}


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CSV_NAME = "output.csv"


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def extract_answer(text: str) -> str:
    """
    Parse model output to find predicted A/B/C/D from patterns like [[A]] or the last letter we see.
    """
    # Quick bracket check
    if "[A]" in text:
        return "A"
    elif "[B]" in text:
        return "B"
    elif "[C]" in text:
        return "C"
    elif "[D]" in text:
        return "D"
    # fallback search from end
    for i in range(len(text) - 1, -1, -1):
        if text[i] in ["A", "B", "C", "D"]:
            return text[i]
    return "A"  # default


def format_prompt_for_test(record, shuffle_choices=True):
    system_msg = SYSTEM_PROMPT
    user_msg, letter_map = _build_user_message(record, shuffle=shuffle_choices)
    return system_msg, user_msg, letter_map


def evaluate_model_on_tom(
    model, tokenizer, subtask_records, subtask_name, device="cuda"
):
    """
    Evaluate a *finetuned or pruned* model on a list of leftover test records.
    Uses separate system and user messages for proper chat formatting.
    """
    if len(subtask_records) == 0:
        return 0.0

    model.eval()
    correct = 0
    for rec in tqdm(
        subtask_records, desc=f"Evaluating on {subtask_name}", unit="record"
    ):
        gold = rec.get("ANSWER\nANSWER", "A") or "A"

        system_msg, user_msg, letter_map = format_prompt_for_test(
            rec, shuffle_choices=True
        )

        # Format as chat messages using the tokenizer's template
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        # Convert to model input format using the tokenizer's chat template
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Convert to tokens
        inputs = tokenizer(chat_text, return_tensors="pt").to(device)
        model = model.to(device)

        # Generate
        out = model.generate(
            **inputs,
            max_length=1024,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
        # Remove the prompt portion
        gen_part = out[0][inputs["input_ids"].shape[1] :]
        gen_text = tokenizer.decode(gen_part, skip_special_tokens=True)
        raw_letter = extract_answer(gen_text)
        mapped_letter = letter_map.get(raw_letter, "A")

        if mapped_letter == gold:
            correct += 1

    return correct / len(subtask_records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-3b-Instruct",
        help="Base model name/path.",
    )
    parser.add_argument(
        "--train_num", type=int, default=32, help="Calibration set size per subtask."
    )
    parser.add_argument(
        "--test_num",
        type=int,
        default=5,
        help="Test set size per subtask, if negative use all.",
    )
    parser.add_argument(
        "--sparsity_ratios",
        nargs="+",
        type=float,
        default=[25, 50, 75],
        help="List of integer percentages for unstructured pruning, e.g. 25 50 75.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    torch.set_default_dtype(torch.float32)
    torch.cuda.empty_cache()

    table_header = ["model_name", "sparsity"] + list(SHORT_NAMES.values())
    results_rows = []

    print("\n=== Evaluating RAW model ===")
    raw_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16
    ).to(DEVICE)
    raw_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    subtask_accs = []
    for eval_task in SHORT_NAMES.keys():
        # purely use leftover test data
        _, test_recs = get_tom(
            tokenizer,
            eval_task,
            train_num=args.train_num,
            test_num=args.test_num,
            seed=args.seed,
        )
        acc = evaluate_model_on_tom(
            raw_model,
            tokenizer,
            test_recs,
            eval_task,
            device=DEVICE,
        )
        subtask_accs.append(acc)

    raw_name = f"RAW_{os.path.basename(args.model)}"
    results_rows.append([raw_name, "0%"] + [f"{x:.4f}" for x in subtask_accs])

    print("\n=== Evaluating PRUNED models ===")

    for i, subtask_file in enumerate(SHORT_NAMES.keys()):
        for ratio in args.sparsity_ratios:
            # 1) Load fresh model
            print(f"\n=== Pruning on subtask '{subtask_file}' at {ratio}% sparsity ===")
            base_model = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=torch.float16
            ).to(DEVICE)

            # 2) Prepare calibration set & test set for *this* subtask
            tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id or 0

            trainloader, _ = get_tom(
                tokenizer,
                subtask_file,
                train_num=args.train_num,
                test_num=0,  # We don't need these leftover test records for the "prune" subtask
                seed=args.seed,
            )

            max_cal_len = 0
            for inp, _, _ in trainloader:
                seq_len = inp.shape[1]  # shape is [1, seq_len]
                if seq_len > max_cal_len:
                    max_cal_len = seq_len

            base_model.seqlen = max_cal_len

            # Actually prune
            llama_sparsellm(
                base_model, trainloader, torch.device(DEVICE), ratio / 100.0
            )

            base_model.eval()

            # Evaluate on all 8 subtasks
            subtask_accs = []
            for eval_task in SHORT_NAMES.keys():
                # get test set from that subtask
                _, test_recs = get_tom(
                    tokenizer,
                    eval_task,
                    train_num=args.train_num,
                    test_num=args.test_num,
                    seed=args.seed,
                )
                # Evaluate
                acc = evaluate_model_on_tom(
                    base_model,
                    tokenizer,
                    test_recs,
                    eval_task,
                    device=DEVICE,
                )
                subtask_accs.append(acc)

            # Build row
            row_model_name = f"Pruned_{os.path.basename(args.model)}_on_{subtask_file}"
            row_sparsity = f"{ratio}%"
            row_vals = [f"{acc:.4f}" for acc in subtask_accs]
            results_rows.append([row_model_name, row_sparsity] + row_vals)

            # Free
            base_model.cpu()
            del base_model
            torch.cuda.empty_cache()

    # Write CSV
    with open(CSV_NAME, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(table_header)
        for row in results_rows:
            writer.writerow(row)

    print(f"\nAll done! Results saved to '{CSV_NAME}'.")
    print("Rows:", len(results_rows))
    print("Columns:", len(table_header))


if __name__ == "__main__":
    main()
