import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from transformers import LogitsProcessorList
from vllm import LLM, SamplingParams

from utils.logits_bias import LogitBiasProcess


COT_TEMPLATE = """You are a professional medical expert answering a multiple-choice medical question.
Please think step by step and then provide exactly one final option.
Format the final answer as: The answer is [LETTER].

Question:
{question}

"""


def extract_option(prediction, model_name=None):
    if not prediction:
        return None

    prediction = prediction.strip()
    patterns = [
        r"The answer is ([A-Z])",
        r"The answer is \$\\boxed{([A-Z])}\$",
        r"The answer is \[([A-Z])\]",
    ]

    for pattern in patterns:
        matches = list(re.finditer(pattern, prediction))
        if matches:
            return matches[-1].group(1)

    if model_name and "gemma" in model_name.lower():
        matches = list(re.finditer(r"The answer is \*\*([A-Z])\*\*", prediction))
        if matches:
            return matches[-1].group(1)

    matches = list(re.finditer(r"The answer is ([A-Z])", prediction, re.IGNORECASE))
    if matches:
        return matches[-1].group(1).upper()

    return None


def load_qa_dataset(file_path, question_type):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if question_type != "mcq":
        return data, None

    allowed_letters = sorted(
        {item["gold"].upper() for item in data if item.get("gold")}
    )
    print(f"Allowed options: {allowed_letters}")
    return data, allowed_letters


def resolve_output_prefix(args):
    if args.output_prefix:
        return Path(args.output_prefix)
    if args.output_file:
        return Path(args.output_file)
    raise ValueError("Either --output_prefix or --output_file must be provided.")


def load_all_existing_results(output_prefix):
    output_dir = output_prefix.parent
    output_stem = output_prefix.name
    gen_files = sorted(
        output_dir.glob(f"{output_stem}_gen*.json"),
        key=lambda path: int(path.stem.rsplit("gen", 1)[1]),
    )

    if not gen_files:
        return []

    all_results = []
    for file_path in gen_files:
        with open(file_path, "r", encoding="utf-8") as f:
            all_results.extend(json.load(f))

    unique_results = []
    seen = set()
    for item in all_results:
        key = (item["id"], item["generation_idx"])
        if key in seen:
            continue
        seen.add(key)
        unique_results.append(item)

    print(f"Loaded {len(unique_results)} records from {len(gen_files)} generation files")
    return unique_results


def get_processed_ids(existing_results):
    return {item["id"] for item in existing_results}


def split_results_by_generation(results):
    grouped = {}
    for item in results:
        generation_idx = item["generation_idx"]
        grouped.setdefault(generation_idx, []).append(item)
    return [grouped[idx] for idx in sorted(grouped)]


def evaluate_generation_group(group):
    golds = [item["gold"].upper() for item in group]
    preds = [(item["pred"] or "UNKNOWN").strip().rstrip(".").upper() for item in group]

    accuracy = round(accuracy_score(golds, preds) * 100, 2)
    recovery_count = sum(1 for item in group if item.get("recovery_used", False))
    invalid_count = sum(1 for pred in preds if pred == "UNKNOWN")

    return {
        "accuracy": accuracy,
        "recovery_used": recovery_count,
        "invalid_answers": invalid_count,
        "sample_count": len(group),
    }


def build_prompt(item, tokenizer, is_base_model):
    if is_base_model:
        return COT_TEMPLATE.format(question=item["question"])

    messages = [
        {
            "role": "system",
            "content": "Please think step by step and output the final answer as 'The answer is [LETTER]'.",
        },
        {"role": "user", "content": item["question"]},
    ]

    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    raise ValueError("Instruct model tokenizer does not provide a chat template.")


def build_recovery_prompt(item, model_answer, tokenizer, is_base_model):
    if is_base_model:
        suffix = "" if model_answer.strip().endswith(".") else "."
        return (
            COT_TEMPLATE.format(question=item["question"])
            + model_answer
            + suffix
            + "\nRespond with one option only.\nThe answer is "
        )

    messages = [
        {
            "role": "system",
            "content": "Please think step by step and output the final answer as 'The answer is [LETTER]'.",
        },
        {"role": "user", "content": item["question"]},
        {"role": "assistant", "content": model_answer},
        {"role": "user", "content": "Respond with one option only. The answer is "},
    ]

    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return (
        COT_TEMPLATE.format(question=item["question"])
        + model_answer
        + "\nRespond with one option only.\nThe answer is "
    )


def build_constrained_sampling_params(args, tokenizer, allowed_letters):
    def add_choice_tokens(choice_letters):
        choice_ids = []
        for letter in choice_letters:
            choice_ids.append(tokenizer.encode(letter)[0])
            if any(
                name in args.model.lower()
                for name in [
                    "gemma",
                    "llama",
                    "biomed",
                    "huatuo",
                    "deepseek-r1",
                    "medreason",
                ]
            ):
                choice_ids.extend(tokenizer.encode(f"{letter}."))
        return list(set(choice_ids))

    logits_processor = LogitsProcessorList()
    logits_processor.append(LogitBiasProcess(add_choice_tokens(allowed_letters)))

    max_tokens = 1 if any(name in args.model for name in ["Qwen3", "Qwen2", "m1"]) else 2
    print(f"Allowed option tokenization: {[(x, tokenizer.encode(x)) for x in allowed_letters]}")

    return SamplingParams(
        temperature=0,
        max_tokens=max_tokens,
        min_tokens=1,
        logits_processors=logits_processor,
    )


def evaluate_model(args):
    if "gpt-oss" in args.model.lower():
        prefix_caching = False
        os.environ["VLLM_USE_V1"] = "1"
    else:
        prefix_caching = True
        os.environ["VLLM_USE_V1"] = "0"

    output_prefix = resolve_output_prefix(args)
    qa_data, allowed_letters = load_qa_dataset(args.dataset, args.question_type)
    print(f"Loaded {len(qa_data)} samples from {args.dataset}")

    existing_results = load_all_existing_results(output_prefix)
    processed_ids = get_processed_ids(existing_results)
    print(f"Found {len(processed_ids)} previously processed samples")

    remaining_data = [item for item in qa_data if item["id"] not in processed_ids]
    if not remaining_data:
        print("No new samples to process.")
        return

    llm_kwargs = {
        "model": args.model,
        "enable_prefix_caching": prefix_caching,
        "max_model_len": args.max_model_len,
        "tensor_parallel_size": max(1, torch.cuda.device_count()),
        "gpu_memory_utilization": 0.9,
    }
    if "gemma-3" in args.model.lower():
        llm_kwargs["enforce_eager"] = True

    llm = LLM(**llm_kwargs)
    tokenizer = llm.llm_engine.tokenizer.tokenizer
    is_base_model = "base" in args.model.lower()
    print(f"Testing a {'base' if is_base_model else 'chat'} model")

    constrained_sampling_params = None
    if args.question_type == "mcq" and allowed_letters:
        constrained_sampling_params = build_constrained_sampling_params(
            args, tokenizer, allowed_letters
        )

    free_sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        n=args.num_generations,
        seed=args.seed if args.seed is not None else None,
    )

    prompts = []
    item_mapping = []
    for item in remaining_data:
        try:
            prompt = build_prompt(item, tokenizer, is_base_model)
        except Exception as exc:
            print(f"Falling back to local prompt template: {exc}")
            prompt = COT_TEMPLATE.format(question=item["question"])
        prompts.append(prompt)
        item_mapping.append(item)

    print(f"Starting batch processing for {len(prompts)} samples")
    free_outputs = llm.generate(prompts, free_sampling_params)

    recovery_data = []
    results = []
    for item, output in zip(item_mapping, free_outputs):
        for generation_idx, generation in enumerate(output.outputs):
            model_answer = generation.text.strip()
            pred_option = (
                extract_option(model_answer, args.model)
                if args.question_type == "mcq"
                else None
            )

            if pred_option or args.question_type != "mcq" or not constrained_sampling_params:
                results.append(
                    {
                        "id": item["id"],
                        "name": item.get("name", "unknown"),
                        "gold": item["gold"],
                        "model": args.model.split("/")[-1],
                        "generation_idx": generation_idx,
                        "answer": model_answer,
                        "pred": pred_option,
                        "recovery_used": False,
                    }
                )
                continue

            recovery_prompt = build_recovery_prompt(
                item, model_answer, tokenizer, is_base_model
            )
            recovery_data.append(
                {
                    "item": item,
                    "generation_idx": generation_idx,
                    "prompt": recovery_prompt,
                    "original_answer": model_answer,
                }
            )

    if recovery_data:
        print(f"Running recovery generation for {len(recovery_data)} samples")
        recovery_prompts = [record["prompt"] for record in recovery_data]
        recovery_outputs = llm.generate(recovery_prompts, constrained_sampling_params)

        for record, output in zip(recovery_data, recovery_outputs):
            recovered_answer = output.outputs[0].text.strip()
            results.append(
                {
                    "id": record["item"]["id"],
                    "name": record["item"].get("name", "unknown"),
                    "gold": record["item"]["gold"],
                    "model": args.model.split("/")[-1],
                    "generation_idx": record["generation_idx"],
                    "answer": record["original_answer"] + "\nThe answer is " + recovered_answer,
                    "pred": recovered_answer,
                    "recovery_used": True,
                }
            )

    final_results = existing_results + results
    results_by_generation = split_results_by_generation(final_results)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for gen_idx, current_results in enumerate(results_by_generation):
        output_path = output_prefix.parent / f"{output_prefix.name}_gen{gen_idx}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(current_results, f, indent=2, ensure_ascii=False)
        print(f"Saved generation {gen_idx} results to {output_path}")

    if args.question_type != "mcq":
        return

    per_generation_metrics = []
    accuracies = []
    for gen_idx, current_group in enumerate(results_by_generation):
        metrics = evaluate_generation_group(current_group)
        accuracies.append(metrics["accuracy"])
        per_generation_metrics.append(
            {
                "generation": gen_idx,
                **metrics,
            }
        )

    log_entry = {
        "dataset": qa_data[0].get("name", "unknown") if qa_data else "unknown",
        "model": args.model.split("/")[-1],
        "per_generation": per_generation_metrics,
        "average_accuracy": round(float(np.mean(accuracies)), 2),
        "std_accuracy": round(float(np.std(accuracies)), 2),
        "num_generations": len(results_by_generation),
        "total_records": len(final_results),
        "params": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
            "num_generations": args.num_generations,
            "seed": args.seed,
        },
    }

    log_path = output_prefix.with_suffix(".eval.log")
    with open(log_path, "a", encoding="utf-8") as f:
        json.dump(log_entry, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps(log_entry, indent=2, ensure_ascii=False))


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model path or HF model id")
    parser.add_argument("--dataset", type=str, required=True, help="Path to a dataset JSON file")
    parser.add_argument(
        "--question_type",
        type=str,
        default="mcq",
        choices=["mcq", "open"],
        help="Question type",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--top_k", type=int, default=-1, help="Top-k sampling")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max generated tokens")
    parser.add_argument("--max_model_len", type=int, default=16000, help="Maximum model length")
    parser.add_argument(
        "--num_generations",
        type=int,
        default=4,
        help="Number of generations per sample",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_prefix",
        type=str,
        default=None,
        help="Prefix for saved files, e.g. ./outputs/medqa_run",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


if __name__ == "__main__":
    evaluate_model(build_parser().parse_args())
