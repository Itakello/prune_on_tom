import argparse
import csv
import os
import random

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from submodules.SparseLLM.datautils import get_tokenizer, get_tom

# Use the pruning/eval routines from the existing SparseLLM code
from submodules.SparseLLM.model_utils import get_llama, llama_sparsellm

############################################################################
# Define the 8 subtask files from ToMBench
############################################################################

TOMBENCH_SUBTASKS = [
    "Ambiguous Story Task.jsonl",
    "False Belief Task.jsonl",
    "Hinting Task Test.jsonl",
    "Faux-pas Recognition Test.jsonl",
    "Unexpected Outcome Test.jsonl",
    "Persuasion Story Task.jsonl",
    "Strange Story Task.jsonl",
    "Scalar Implicature Task.jsonl",
]

# For CSV column naming
SHORT_NAMES = {
    "Ambiguous Story Task.jsonl": "Ambiguous",
    "False Belief Task.jsonl": "FalseBelief",
    "Hinting Task Test.jsonl": "Hinting",
    "Faux-pas Recognition Test.jsonl": "FauxPas",
    "Unexpected Outcome Test.jsonl": "UnexpOut",
    "Persuasion Story Task.jsonl": "Persuasion",
    "Strange Story Task.jsonl": "Strange",
    "Scalar Implicature Task.jsonl": "ScalarImp",
}

############################################################################
# Simple evaluation logic for test records
############################################################################


def extract_answer(text: str) -> str:
    """
    Parse model output to find predicted A/B/C/D from patterns like [[A]] or the last letter we see.
    """
    # Quick bracket check
    if "[[A]]" in text:
        return "A"
    elif "[[B]]" in text:
        return "B"
    elif "[[C]]" in text:
        return "C"
    elif "[[D]]" in text:
        return "D"
    # fallback search from end
    for i in range(len(text) - 1, -1, -1):
        if text[i] in ["A", "B", "C", "D"]:
            return text[i]
    return "A"  # default


def most_common(lst):
    counts = {}
    for x in lst:
        counts[x] = counts.get(x, 0) + 1
    return max(counts, key=counts.get)


def format_prompt_for_test(record, shuffle_choices=True):
    """
    Build a [SYSTEM] + [USER] style prompt for the test record,
    but do *not* reveal the correct answer. We'll randomize the letter ordering for multiple tries.
    """
    system_prompt = """Below is a multiple-choice question with a story and several answer options. Based on the content of the story and the given question, please infer the most likely answer and output the answer index.
    Note:
    (1) Please only output the most likely answer index in the format: [[Answer Index]];
    (2) You must choose one of A, B, C, D even if the story doesn't have enough info;
    (3) Output only the answer index, nothing else.
    """

    # We'll do a plain prefix: "<s>[SYSTEM]\n..."
    # Then [USER] portion includes the story, question, and the candidate answers in random order if desired.
    # Distinguish 2-choice vs 4-choice
    optC = record.get("OPTION-C", None)
    if optC is not None:
        # 4-choice
        A = record["OPTION-A"].replace("A. ", "")
        B = record["OPTION-B"].replace("B. ", "")
        C = record["OPTION-C"].replace("C. ", "")
        D = record["OPTION-D"].replace("D. ", "")
        all_choices = [("A", A), ("B", B), ("C", C), ("D", D)]
    else:
        # 2-choice
        A = record["OPTION-A"].replace("A. ", "")
        B = record["OPTION-B"].replace("B. ", "")
        all_choices = [("A", A), ("B", B)]

    if shuffle_choices:
        random.shuffle(all_choices)

    user_part = (
        f"[Story]\n{record['STORY']}\n\n"
        f"[Question]\n{record['QUESTION']}\n\n"
        f"[Candidate Answers]\n"
    )
    letter_map = {}
    for i, (orig_letter, textval) in enumerate(all_choices):
        new_letter = chr(ord("A") + i)  # 'A' or 'B' or ...
        user_part += f"{new_letter}. {textval}\n"
        # We'll remember how new_letter maps to the original "A/B/C/D"
        letter_map[new_letter] = orig_letter

    system_block = "[SYSTEM]\n" + system_prompt
    user_block = "[USER]\n" + user_part
    # Final text
    return system_block + "\n" + user_block, letter_map


def evaluate_model_on_tom(
    model, tokenizer, subtask_records, subtask_name, try_times=3, device="cuda"
):
    """
    Evaluate a *finetuned or pruned* model on a list of leftover test records (like the output from get_tom).
    We do 'try_times' random shuffles for each record, then majority-vote the predicted letter.
    Return accuracy float in [0..1].
    """
    if len(subtask_records) == 0:
        return 0.0

    model.eval()
    correct = 0
    for rec in tqdm(
        subtask_records, desc=f"Evaluating on {subtask_name}", unit="record"
    ):
        gold = rec.get("ANSWER\nANSWER", "A") or "A"

        these_preds = []
        for _ in range(try_times):
            prompt_text, letter_map = format_prompt_for_test(rec, shuffle_choices=True)
            # Convert to tokens
            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
            model = model.to(device)
            # Generate
            out = model.generate(
                **inputs,
                max_length=1024,
                do_sample=True,
                top_k=1,
                pad_token_id=tokenizer.pad_token_id,
            )
            # Remove the prompt portion
            gen_part = out[0][inputs["input_ids"].shape[1] :]
            gen_text = tokenizer.decode(gen_part, skip_special_tokens=True)

            raw_letter = extract_answer(gen_text)
            mapped_letter = letter_map.get(raw_letter, "A")
            these_preds.append(mapped_letter)

        # majority vote
        maj = most_common(these_preds)
        if maj == gold:
            correct += 1

    return correct / len(subtask_records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-3b-Instruct",
        help="Base model name/path.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--train_num", type=int, default=32, help="Calibration set size per subtask."
    )
    parser.add_argument(
        "--test_num", type=int, default=5, help="Test set size per subtask."
    )
    parser.add_argument(
        "--try_times",
        type=int,
        default=3,
        help="How many random shuffles per test sample.",
    )
    parser.add_argument("--output_csv", type=str, default="results.csv")
    parser.add_argument(
        "--sparsity_ratios",
        nargs="+",
        type=float,
        default=[25, 50, 75],
        help="List of integer percentages for unstructured pruning, e.g. 25 50 75.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.set_default_dtype(torch.float32)
    torch.cuda.empty_cache()

    # We will create a table with columns:
    # [model_name, sparsity, <subtask1>, <subtask2>, ..., <subtask8>]
    # We'll store them in memory and write at the end.

    table_header = ["model_name", "sparsity"] + [
        SHORT_NAMES[s] for s in TOMBENCH_SUBTASKS
    ]
    results_rows = []

    print("\n=== Evaluating RAW model ===")
    raw_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16
    ).to(args.device)
    raw_model.eval()
    tokenizer = get_tokenizer(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    subtask_accs = []
    for eval_task in TOMBENCH_SUBTASKS:
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
            try_times=args.try_times,
            device=args.device,
        )
        subtask_accs.append(acc)

    raw_name = f"RAW_{os.path.basename(args.model)}"
    results_rows.append([raw_name, "0%"] + [f"{x:.4f}" for x in subtask_accs])

    for i, subtask_file in enumerate(TOMBENCH_SUBTASKS):
        for ratio in args.sparsity_ratios:
            # 1) Load fresh model
            print(f"\n=== Pruning on subtask '{subtask_file}' at {ratio}% sparsity ===")
            base_model = get_llama(args).to(args.device)
            base_model.eval()

            # 2) Prepare calibration set & test set for *this* subtask
            tokenizer = get_tokenizer(args.model)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id or 0

            trainloader, _ = get_tom(
                tokenizer,
                subtask_file,
                train_num=args.train_num,
                test_num=0,  # We don't need these leftover test records for the "prune" subtask
                seed=args.seed,
            )

            # Build the ephemeral pruning args
            pargs = argparse.Namespace()
            pargs.sparsity = ratio / 100.0  # unstructured fraction
            pargs.prunen = 0
            pargs.prunem = 0
            pargs.percdamp = 0.01
            pargs.blocksize = 128
            pargs.gmp = False
            pargs.wbits = 16
            pargs.minlayer = -1
            pargs.maxlayer = 1000
            pargs.prune_only = ""
            pargs.invert = False
            pargs.save = ""
            pargs.true_sequential = False
            pargs.log_wandb = False
            pargs.nsamples = len(trainloader)

            max_cal_len = 0
            for inp, _, _ in trainloader:
                seq_len = inp.shape[1]  # shape is [1, seq_len]
                if seq_len > max_cal_len:
                    max_cal_len = seq_len

            base_model.seqlen = max_cal_len

            # Actually prune
            llama_sparsellm(
                base_model, tokenizer, trainloader, torch.device(args.device), pargs
            )

            # Evaluate on all 8 subtasks
            subtask_accs = []
            for eval_task in TOMBENCH_SUBTASKS:
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
                    try_times=args.try_times,
                    device=args.device,
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
    with open(args.output_csv, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(table_header)
        for row in results_rows:
            writer.writerow(row)

    print(f"\nAll done! Results saved to '{args.output_csv}'.")
    print("Rows:", len(results_rows))
    print("Columns:", len(table_header))


if __name__ == "__main__":
    main()
