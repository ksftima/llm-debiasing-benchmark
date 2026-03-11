"""
Step 1 of 2: Annotate a parsed dataset CSV using a local HuggingFace model.

Runs inference on a GPU (or CPU as fallback) using the transformers pipeline.
Saves one prompt and one response .txt file per row, same format as annotate_api.py,
so gather_responses.py works identically afterwards.

Run on Vera GPU nodes via thesis/exe/vera-apptainer/annotate-local-model.sh.

Usage:
    python annotate_local_model.py <dataset> <parsed_csv> <annotation_dir>
        --model <model_path_or_hf_id>
        [--num N] [--start S] [--num_examples K] [--batchsize B]

Arguments:
    dataset         Dataset name: fomc | pubmedqa | cuad | misogynistic
    parsed_csv      Path to the parsed scaled CSV file
    annotation_dir  Directory where prompts/responses will be saved

Options:
    --model         HuggingFace model ID or local path (default: microsoft/phi-4)
    --num           Number of rows to annotate (default: all)
    --start         Row offset to start from (default: 0)
    --num_examples  Number of few-shot examples, one per class (default: 0)
    --batchsize     Number of rows to process per GPU batch (default: 8)
"""

import os
import argparse
from pathlib import Path

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from annotate_prompts import system_prompts, dataset_labels, make_user_prompt


####################
# File I/O helpers #
####################

def save_prompt(annotation_dir, prompt, index):
    path = annotation_dir / "prompts"
    path.mkdir(parents=True, exist_ok=True)
    (path / f"prompt_{index:05}.txt").write_text(prompt, encoding="utf-8")


def save_response(annotation_dir, response, index):
    path = annotation_dir / "responses"
    path.mkdir(parents=True, exist_ok=True)
    (path / f"response_{index:05}.txt").write_text(response, encoding="utf-8")


def save_error(annotation_dir, error, index):
    path = annotation_dir / "errors"
    path.mkdir(parents=True, exist_ok=True)
    (path / f"error_{index:05}.txt").write_text(error, encoding="utf-8")


########
# Main #
########

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Annotate a parsed CSV using a local model.")
    parser.add_argument("dataset", choices=list(system_prompts.keys()),
                        help="Dataset name")
    parser.add_argument("parsed_csv", type=Path,
                        help="Path to parsed scaled CSV file")
    parser.add_argument("annotation_dir", type=Path,
                        help="Directory to save prompts/responses into")
    parser.add_argument("--model", type=str, default="microsoft/phi-4",
                        help="HuggingFace model ID or local path (default: microsoft/phi-4)")
    parser.add_argument("--num", type=int, default=None,
                        help="Number of rows to annotate (default: all)")
    parser.add_argument("--start", type=int, default=0,
                        help="Row index to start from (default: 0)")
    parser.add_argument("--num_examples", type=int, default=0,
                        help="Number of few-shot examples, stratified by class (default: 0)")
    parser.add_argument("--batchsize", type=int, default=8,
                        help="GPU batch size (default: 8)")
    args = parser.parse_args()

    # Device selection
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    # Load data — only text and y columns
    data = pd.read_csv(args.parsed_csv, usecols=["text", "y"])
    data = data.iloc[args.start: args.start + args.num] if args.num else data.iloc[args.start:]
    data = data.reset_index(drop=True)

    # Stratified few-shot examples (one per class, then fill remaining randomly)
    examples = []
    if args.num_examples > 0:
        labels = dataset_labels[args.dataset]
        classes = sorted(data["y"].unique())
        example_rows = pd.concat([
            data[data["y"] == c].sample(1) for c in classes
        ])
        remaining = args.num_examples - len(classes)
        if remaining > 0:
            pool = data.loc[~data.index.isin(example_rows.index)]
            extra = pool.sample(min(remaining, len(pool)))
            example_rows = pd.concat([example_rows, extra])
        data = data.loc[~data.index.isin(example_rows.index)].reset_index(drop=True)
        examples = [(row["text"], labels[int(row["y"])]) for _, row in example_rows.iterrows()]

    print(f"Annotating {len(data)} rows with {args.model} ({args.dataset})")
    print(f"Few-shot examples: {len(examples)}")
    print(f"Output directory: {args.annotation_dir}\n")

    args.annotation_dir.mkdir(parents=True, exist_ok=True)

    # Load model and tokenizer
    print(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"Loading model for {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=device,
        torch_dtype=torch.float16,
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=10,
        num_return_sequences=1,
    )

    system_prompt = system_prompts[args.dataset]

    def make_chat(text):
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": make_user_prompt(args.dataset, text, examples if examples else None)},
        ]

    # Run inference in batches
    from itertools import batched
    total = len(data)
    done = 0

    for batch_indices in batched(data.index, args.batchsize):
        batch_indices = list(batch_indices)
        chats = [make_chat(data["text"][i]) for i in batch_indices]

        try:
            outputs = generator(chats)
        except Exception as e:
            for i in batch_indices:
                save_error(args.annotation_dir, str(e), i)
            print(f"  Batch {batch_indices[0]:05}-{batch_indices[-1]:05} failed: {e}")
            continue

        for i, output in zip(batch_indices, outputs):
            # The last user message is at index -2, model response at -1
            prompt = output[0]["generated_text"][-2]["content"]
            response = output[0]["generated_text"][-1]["content"]
            save_prompt(args.annotation_dir, prompt, i)
            save_response(args.annotation_dir, response, i)

        done += len(batch_indices)
        print(f"  {done}/{total} done")

    print(f"\nDone. Run gather_responses.py to assemble the annotated CSV.")
