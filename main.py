"""
main.py

Implements project Steps 2 & 3:

 - For each of the 8 ToMBench subtasks:
   (a) sample some "train" examples to prune on (e.g., 20)
   (b) prune the raw model 3 ways (unstructured, 2:4, 4:8)
   (c) evaluate each pruned model on all 8 subtasks
 - Also evaluate the raw model on all 8 subtasks (once) so we have a baseline

We remove the original Step 1 (the large "full dataset" eval),
and allow the user to specify how many samples to evaluate on
(fixed number or "leftover" from the training set).
"""

import argparse
import csv
import json
import os
import random

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Wanda / pruning imports
from submodules.wanda.lib.prune import (
    check_sparsity,
    prepare_calibration_input,
    prune_wanda,
)

# --------------------------------------------------------------------------
# ------------------- 1. TOMBENCH SUBTASKS / CONSTANTS ---------------------
# --------------------------------------------------------------------------

TOMBENCH_SUBTASKS = [
    "Ambiguous Story Task.jsonl",
    "False Belief Task.jsonl",
    "Hinting Test.jsonl",
    "Faux-pas Recognition Test.jsonl",
    "Unexpected Outcome Test.jsonl",
    "Persuasion Story Task.jsonl",
    "Strange Story Task.jsonl",
    "Scalar Implicature Task.jsonl",
]

# Sane short code-names if you wish
TASK_ACRONYMS = {
    "Ambiguous Story Task.jsonl": "AST",
    "False Belief Task.jsonl": "FBT",
    "Hinting Task Test.jsonl": "HT",
    "Faux-pas Recognition Test.jsonl": "FPT",
    "Unexpected Outcome Test.jsonl": "UOT",
    "Persuasion Story Task.jsonl": "PST",
    "Strange Story Task.jsonl": "SST",
    "Scalar Implicature Task.jsonl": "SIT",
}

# The raw model name
RAW_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"  # adjust if needed

# Three sparsity “types” => (prune_n, prune_m) pairs
SPARSITY_MODES = {
    "unstructured": (0, 0),  # Wanda unstructured => (n=0, m=0)
    "2:4": (2, 4),
    "4:8": (4, 8),
}

# Simple system/user prompts for evaluation
SystemEvaluatePrompt_en = """Below is a multiple-choice question with a story and several answer options. Based on the content of the story and the given question, please infer the most likely answer and output the answer index.
Note:
(1) Please only output the most likely answer index in the format: [[Answer Index]];
(2) You must choose one of A, B, C, D even if the story doesn't have enough info;
(3) Output only the answer index, nothing else.
"""

UserEvaluatePrompt4Choices_en = """[Story]
{story}

[Question]
{question}

[Candidate Answers]
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}"""

UserEvaluatePrompt2Choices_en = """[Story]
{story}

[Question]
{question}

[Candidate Answers]
A. {choice_a}
B. {choice_b}"""


# --------------------------------------------------------------------------
# ------------------- 2. HELPER FUNCTIONS FOR PROMPTS ----------------------
# --------------------------------------------------------------------------


def extract_answer(text: str) -> str:
    """
    Parse model output to find predicted A/B/C/D from patterns like [[A]] or just the last letter.
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
    # fallback: search from end
    for i in range(len(text) - 1, -1, -1):
        if text[i] in ["A", "B", "C", "D"]:
            return text[i]
    return "A"  # default fallback


def most_common_element(lst):
    freq = {}
    for x in lst:
        freq[x] = freq.get(x, 0) + 1
    # Return whichever key has largest freq
    return max(freq, key=freq.get)


def format_prompt_en_4choices(sample_dict):
    """
    For 4-choice items in English, random shuffle the 4 answers
    so each call to evaluate can do multiple tries with random choice ordering.
    """
    cA = sample_dict["OPTION-A"].replace("A. ", "")
    cB = sample_dict["OPTION-B"].replace("B. ", "")
    cC = sample_dict["OPTION-C"].replace("C. ", "")
    cD = sample_dict["OPTION-D"].replace("D. ", "")
    choices = [cA, cB, cC, cD]
    random.shuffle(choices)

    user_prompt = UserEvaluatePrompt4Choices_en.format(
        story=sample_dict["STORY"],
        question=sample_dict["QUESTION"],
        choice_a=choices[0],
        choice_b=choices[1],
        choice_c=choices[2],
        choice_d=choices[3],
    )

    # We must map the predicted letter “A/B/C/D” back to the canonical answer
    letter_map = {"A": "", "B": "", "C": "", "D": ""}
    for i, txt in enumerate(choices):
        if txt == cA:
            letter_map[chr(ord("A") + i)] = "A"
        elif txt == cB:
            letter_map[chr(ord("A") + i)] = "B"
        elif txt == cC:
            letter_map[chr(ord("A") + i)] = "C"
        elif txt == cD:
            letter_map[chr(ord("A") + i)] = "D"

    return letter_map, user_prompt


def format_prompt_en_2choices(sample_dict):
    """
    For 2-choice items, random shuffle the 2 answers.
    """
    cA = sample_dict["OPTION-A"].replace("A. ", "")
    cB = sample_dict["OPTION-B"].replace("B. ", "")
    choices = [cA, cB]
    random.shuffle(choices)

    user_prompt = UserEvaluatePrompt2Choices_en.format(
        story=sample_dict["STORY"],
        question=sample_dict["QUESTION"],
        choice_a=choices[0],
        choice_b=choices[1],
    )
    letter_map = {"A": "", "B": "", "C": "", "D": ""}

    if choices[0] == cA:
        letter_map["A"] = "A"
    else:
        letter_map["A"] = "B"

    if choices[1] == cA:
        letter_map["B"] = "A"
    else:
        letter_map["B"] = "B"

    return letter_map, user_prompt


# --------------------------------------------------------------------------
# 3. BUILD PRUNING / TRAINING SAMPLES FOR WANDA
# --------------------------------------------------------------------------


def build_calibration_prompt(sample_dict):
    """
    Single text block that includes correct answer for Wanda's calibration.
    [SYSTEM] <system_prompt>
    [USER]   <all choices in canonical order>
    [ASSISTANT]  [[CORRECT LETTER]]

    No random shuffle here. We keep the correct letter in canonical position.
    """
    # system
    system_part = SystemEvaluatePrompt_en

    # Are we 2-choice or 4-choice?
    is_4choice = sample_dict.get("OPTION-C", None) is not None

    if is_4choice:
        A = sample_dict["OPTION-A"].replace("A. ", "")
        B = sample_dict["OPTION-B"].replace("B. ", "")
        C = sample_dict["OPTION-C"].replace("C. ", "")
        D = sample_dict["OPTION-D"].replace("D. ", "")
        user_part = (
            f"[Story]\n{sample_dict['STORY']}\n\n"
            f"[Question]\n{sample_dict['QUESTION']}\n\n"
            f"[Candidate Answers]\n"
            f"A. {A}\n"
            f"B. {B}\n"
            f"C. {C}\n"
            f"D. {D}"
        )
    else:
        A = sample_dict["OPTION-A"].replace("A. ", "")
        B = sample_dict["OPTION-B"].replace("B. ", "")
        user_part = (
            f"[Story]\n{sample_dict['STORY']}\n\n"
            f"[Question]\n{sample_dict['QUESTION']}\n\n"
            f"[Candidate Answers]\n"
            f"A. {A}\n"
            f"B. {B}"
        )

    correct_ans = sample_dict.get("ANSWER\nANSWER", "A") or "A"
    assistant_part = f"[[{correct_ans}]]"

    text_block = (
        f"[SYSTEM]\n{system_part}\n\n"
        f"[USER]\n{user_part}\n\n"
        f"[ASSISTANT]\n{assistant_part}"
    )
    return text_block


def split_subtask_data(records, train_num=20, seed=42):
    """
    Shuffle subtask data. Return (train_records, leftover_records)
      - train_records => used for Wanda calibration
      - leftover_records => the rest for possible evaluation
    """
    random.seed(seed)
    random.shuffle(records)
    train = records[:train_num]
    leftover = records[train_num:]
    return train, leftover


def get_subtask_data(subtask_file):
    """
    Load the entire subtask's JSON lines from ToMBench.
    """
    path = os.path.join("submodules", "ToMBench", "data", subtask_file)
    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    return records


def tokenize_calibration_data(tokenizer, text_blocks):
    """
    Tokenize a list of strings, then pad them to the max length among them.
    Returns a list of dicts each with 'input_ids' and 'attention_mask'.
    """
    tokenized_list = []
    max_len = 0
    for txt in text_blocks:
        enc = tokenizer(txt, add_special_tokens=False)
        length = len(enc["input_ids"])
        if length > max_len:
            max_len = length
        tokenized_list.append(enc)

    # Now pad each to max_len
    for enc in tokenized_list:
        needed = max_len - len(enc["input_ids"])
        if needed > 0:
            enc["input_ids"] += [tokenizer.pad_token_id] * needed
            enc["attention_mask"] += [0] * needed

    return tokenized_list


class WandaCalibrationDataset(Dataset):
    """
    Simple dataset for Wanda calibration:
     - item[i] => (input_ids, target_ids) with target_ids having -100 for all but next-token
    """

    def __init__(self, enc_list):
        self.enc_list = enc_list

    def __len__(self):
        return len(self.enc_list)

    def __getitem__(self, idx):
        enc = self.enc_list[idx]
        x = torch.tensor(enc["input_ids"], dtype=torch.long)
        # typical next-token: shift by 1 => ignoring the last token
        y = x.clone()
        if y.size(0) > 1:
            y[:-1] = -100
        else:
            y[:] = -100

        # Return batch dimension of 1
        return x.unsqueeze(0), y.unsqueeze(0)


# --------------------------------------------------------------------------
# 4. EVALUATION: Evaluate a given model on a subtask
# --------------------------------------------------------------------------


def evaluate_on_subtask(
    model,
    tokenizer,
    subtask_file,
    try_times=3,
    eval_num=None,
    skip_first_n=0,
    device="cuda",
):
    """
    Evaluate `model` on `subtask_file`.
      - If `eval_num` is not None, we sample up to that many from the data (after skipping skip_first_n).
      - If skip_first_n>0, we remove that many from the front (this can be the "training" portion).
      - We do majority-vote across 'try_times' random orderings.
    Returns an accuracy in [0..1].
    """
    records = get_subtask_data(subtask_file)
    # skip training portion
    records = records[skip_first_n:]

    if eval_num is not None and eval_num > 0:
        # sample or slice
        records = records[:eval_num]

    if len(records) == 0:
        return 0.0  # no data => 0 or could skip

    system_prompt = SystemEvaluatePrompt_en
    has_chat_method = hasattr(tokenizer, "apply_chat_template")

    model.eval()
    model.to(device)

    preds_for_sample = [[] for _ in range(len(records))]

    for i, sample in tqdm(enumerate(records), desc=f"Evaluating {subtask_file}"):
        # check 2-choice or 4-choice
        is_4choice = sample.get("OPTION-C", None) is not None
        for _t in range(try_times):
            if is_4choice:
                letter_map, user_prompt = format_prompt_en_4choices(sample)
            else:
                letter_map, user_prompt = format_prompt_en_2choices(sample)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            if has_chat_method:
                inputs = tokenizer.apply_chat_template(
                    messages, return_tensors="pt", tokenize=True, return_dict=True
                ).to(device)
            else:
                # fallback plain text
                text = f"<s>[SYSTEM]\n{system_prompt}\n[USER]\n{user_prompt}"
                inputs = tokenizer(text, return_tensors="pt").to(device)

            gen_kwargs = dict(
                max_length=1024,
                do_sample=True,
                top_k=1,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
            input_len = inputs["input_ids"].shape[1]
            gen_text = tokenizer.decode(
                outputs[0][input_len:], skip_special_tokens=True
            )

            pred_letter = extract_answer(gen_text)
            mapped_letter = letter_map.get(pred_letter, "A")
            preds_for_sample[i].append(mapped_letter)

    # majority vote
    correct = 0
    for i, sample in enumerate(records):
        gold = sample.get("ANSWER\nANSWER", "A") or "A"
        maj = most_common_element(preds_for_sample[i])
        if maj == gold:
            correct += 1
    return correct / max(1, len(records))


# --------------------------------------------------------------------------
# 5. MAIN: create 24 pruned models & evaluate, plus raw model baseline
# --------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train_num",
        type=int,
        default=32,
        help="Number of samples used to prune on each subtask",
    )
    parser.add_argument(
        "--try_times",
        type=int,
        default=3,
        help="Number of multiple tries for majority vote",
    )
    parser.add_argument(
        "--sparsity_ratio",
        type=float,
        default=0.5,
        help="Global fraction of weights to prune for unstructured or 2:4/4:8 patterns",
    )
    parser.add_argument(
        "--eval_type",
        type=str,
        choices=["remaining", "fixed"],
        default="fixed",
        help="If 'remaining', we skip the training samples for that subtask. If 'fixed', we evaluate on a fixed number of items from each subtask.",
    )
    parser.add_argument(
        "--eval_num",
        type=int,
        default=20,
        help="How many items to evaluate if eval_type='fixed'. If eval_type='remaining', this is ignored for the prune subtask, but used for other tasks.",
    )
    parser.add_argument("--output_csv", type=str, default="results.csv")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load the raw model + tokenizer
    print(f"Loading raw model: {RAW_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(RAW_MODEL_NAME, trust_remote_code=True)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    """model = AutoModelForCausalLM.from_pretrained(
        RAW_MODEL_NAME,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        trust_remote_code=True,
    )

    # 1) Evaluate the raw model on all 8 subtasks
    #    This gives us the baseline lines in the CSV
    print("Evaluating the raw model on all subtasks...")
    with open(args.output_csv, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["model_name", "prune_subtask", "eval_subtask", "accuracy"])

        raw_baseline_name = "RawModel"

        for eval_task in TOMBENCH_SUBTASKS:
            if args.eval_type == "fixed":
                # Evaluate on first 'eval_num' samples
                skip_n = 0
                used_eval_num = args.eval_num
            else:
                # 'remaining' doesn't apply for raw model, so we do the same as fixed
                skip_n = 0
                used_eval_num = args.eval_num

            acc = evaluate_on_subtask(
                model,
                tokenizer,
                eval_task,
                try_times=args.try_times,
                eval_num=used_eval_num,
                skip_first_n=skip_n,
                device=device,
            )
            writer.writerow([raw_baseline_name, "None", eval_task, f"{acc:.4f}"])

        fout.flush()

    # Free (then re-load) to ensure each new pass starts from a fresh raw model
    model.cpu()
    del model
    torch.cuda.empty_cache()"""

    # 2) For each subtask => create 3 pruned LLMs => evaluate them on all subtasks
    for subtask_file in TOMBENCH_SUBTASKS:
        acronym = TASK_ACRONYMS.get(subtask_file, subtask_file)
        print(f"\n=== Pruning subtask = {subtask_file} ({acronym}) ===")

        # Load raw model again
        model = AutoModelForCausalLM.from_pretrained(
            RAW_MODEL_NAME,
            torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
            trust_remote_code=True,
        )
        model.eval()

        # Load the subtask data
        full_records = get_subtask_data(subtask_file)
        train_records, leftover_records = split_subtask_data(
            full_records, train_num=args.train_num, seed=args.seed
        )
        # Build calibration prompts
        train_texts = [build_calibration_prompt(r) for r in train_records]
        # Tokenize
        train_encs = tokenize_calibration_data(tokenizer, train_texts)
        train_ds = WandaCalibrationDataset(train_encs)
        trainloader = DataLoader(train_ds, batch_size=1, shuffle=False)

        # Prepare Wanda calibration input
        with torch.no_grad():
            inps, outs, attention_mask, position_ids = prepare_calibration_input(
                model, trainloader, device
            )

        # For each of the 3 sparsity structures => prune => evaluate => log
        for sp_name, (pn, pm) in SPARSITY_MODES.items():
            print(
                f"  -> Pruning with {sp_name}, n={pn}, m={pm}, ratio={args.sparsity_ratio}"
            )
            # We call the relevant prune method (e.g. Wanda, magnitude, etc.).
            # Here we demonstrate Wanda only; adapt as needed:
            # prune_magnitude() / prune_sparsegpt() / ...
            prune_wanda_args = argparse.Namespace(
                nsamples=len(train_ds),
                seed=args.seed,
                sparsity_ratio=args.sparsity_ratio,
                use_variant=False,  # or True if you want the Wanda variant
                prune_method="wanda",  # for consistent logging, etc.
            )
            # Actually prune
            prune_wanda(
                prune_wanda_args, model, tokenizer, device, prune_n=pn, prune_m=pm
            )
            # Check final sparsity
            sp = check_sparsity(model)
            print(f"     Pruned => actual global sparsity: {sp:.4f}")

            # Evaluate on all tasks => write to CSV
            with open(args.output_csv, "a", newline="", encoding="utf-8") as fout:
                writer = csv.writer(fout)

                # We’ll name this model (subtask=..., sp_name=...)
                model_name_str = f"Pruned_{acronym}_{sp_name}"

                for eval_task in TOMBENCH_SUBTASKS:
                    if args.eval_type == "remaining" and (eval_task == subtask_file):
                        # skip exactly the train_num from the front
                        skip_n = args.train_num
                        used_eval_num = None  # means “use all leftover”
                    elif args.eval_type == "fixed":
                        skip_n = 0
                        used_eval_num = args.eval_num
                    else:
                        # 'remaining' for a different subtask => we just use the entire dataset or a fixed #?
                        # Typically we might do the entire set. If you want a partial, set eval_num or adapt logic.
                        used_eval_num = args.eval_num
                        skip_n = 0

                    acc = evaluate_on_subtask(
                        model,
                        tokenizer,
                        eval_task,
                        try_times=args.try_times,
                        eval_num=used_eval_num,
                        skip_first_n=skip_n,
                        device=device,
                    )
                    writer.writerow(
                        [model_name_str, subtask_file, eval_task, f"{acc:.4f}"]
                    )
                fout.flush()

            # (Optionally) revert to unpruned raw for next loop => but we’re only
            # building 1 pruned model per loop. So we must RELOAD raw model each time
            # if we want a fresh start. We do that above at the subtask for-loop start.
            # So after we finish one sparsity type, we reload raw again—if we want
            # to do “clean” pruning. For memory reasons you can keep multiple copies,
            # but simpler to reload:

            # Move pruned model CPU to free GPU memory
            model.cpu()
            del model
            torch.cuda.empty_cache()

            # Reload raw to prune the next type
            model = AutoModelForCausalLM.from_pretrained(
                RAW_MODEL_NAME,
                torch_dtype=(
                    torch.float16 if device.startswith("cuda") else torch.float32
                ),
                trust_remote_code=True,
            )
            model.eval()
            # We must re-generate the Wanda calibration inps for each pass:
            with torch.no_grad():
                inps, outs, attention_mask, position_ids = prepare_calibration_input(
                    model, trainloader, device
                )

        # Done with the 3 prunes for this subtask
        model.cpu()
        del model
        torch.cuda.empty_cache()

    print("\nAll done! You now have a CSV containing:")
    print(" - Raw model results on all subtasks")
    print(
        " - 24 pruned models (8 subtasks x 3 sparsity types) each evaluated on all 8 subtasks."
    )
    print("Check:", args.output_csv)


if __name__ == "__main__":
    main()
