"""
Step 1 of 2: Annotate a parsed dataset CSV using DeepSeek or OpenAI.

Reads API keys from .env in the repo root.
Saves one prompt and one response .txt file per row into:
    <annotation_dir>/prompts/prompt_XXXXX.txt
    <annotation_dir>/responses/response_XXXXX.txt
    <annotation_dir>/errors/error_XXXXX.txt  (on failure)

Run gather_responses.py afterwards to assemble the annotated CSV.

Usage:
    python annotate_api.py <api> <dataset> <parsed_csv> <annotation_dir>
        [--num N] [--start S] [--num_examples K]

Arguments:
    api             Which API to use: "deepseek" or "openai"
    dataset         Dataset name: fomc | pubmedqa | cuad | misogynistic
    parsed_csv      Path to the parsed scaled CSV file
    annotation_dir  Directory where prompts/responses will be saved

Options:
    --num           Number of rows to annotate (default: all)
    --start         Row offset to start from (default: 0)
    --num_examples  Number of few-shot examples to include in prompts (default: 0)

How batching works:
    DeepSeek: does not have a native batch API, so requests are sent in
        parallel using a thread pool (200 at a time). This is cheaper and
        faster than sequential calls but uses the same per-token pricing.
    OpenAI: uses the official Batch API which costs 50% less than real-time
        calls. Requests are written to a JSONL file, uploaded, and a batch
        job is created. The script polls until the job is complete (up to 24h).

How the LLM is prevented from seeing expert labels (y):
    Only the "text" column is passed to make_user_prompt(). The y column and
    all x features stay in the dataframe on disk and are never included in
    any prompt or API call.
"""

import os
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import batched
import argparse

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from annotate_prompts import dataset_labels, system_prompts, make_user_prompt


# Load API keys from .env in the repo root
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")


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


############
# DeepSeek #
############
# DeepSeek exposes an OpenAI-compatible API but has no native batch endpoint.
# We send up to 200 requests in parallel using a thread pool, which is the
# recommended approach for throughput. Each thread makes one synchronous call.

def _call_deepseek_single(client, system_prompt, user_prompt):
    """Make a single DeepSeek chat completion call. Returns the response text."""
    api_response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
        max_tokens=10,
    )
    return api_response.choices[0].message.content


def annotate_deepseek(system_prompt, user_prompts, annotation_dir):
    """
    Annotate a dict of {index: prompt} using DeepSeek in parallel batches.

    Sends up to 200 requests concurrently per batch to stay within rate limits.
    Results are saved to disk as they complete so partial progress is not lost.
    """
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )

    for batch_indices in batched(user_prompts.keys(), 200):
        batch_indices = list(batch_indices)
        print(f"Sending batch rows {batch_indices[0]:05} – {batch_indices[-1]:05}")

        futures = {}
        with ThreadPoolExecutor(max_workers=len(batch_indices)) as executor:
            for index in batch_indices:
                future = executor.submit(
                    _call_deepseek_single, client, system_prompt, user_prompts[index]
                )
                futures[future] = index

            for future in as_completed(futures):
                index = futures[future]
                try:
                    response = future.result()
                    save_prompt(annotation_dir, user_prompts[index], index)
                    save_response(annotation_dir, response, index)
                except Exception as e:
                    save_error(annotation_dir, str(e), index)
                    print(f"  Row {index:05} failed: {e}")


##########
# OpenAI #
##########
# OpenAI's Batch API charges 50% of the standard per-token price.
# We upload a JSONL file of requests, create a batch job, then poll until done.
# The batch can take up to 24h but typically finishes in minutes to hours.

def annotate_openai(system_prompt, user_prompts, annotation_dir):
    """
    Annotate a dict of {index: prompt} using the OpenAI Batch API.

    Writes the batch input to input_file.jsonl and saves the batch ID to
    batch_id.txt so you can retrieve results manually if the script is
    interrupted. Output is saved to output_file.jsonl and individual
    response txt files.
    """
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Build one request object per row
    def make_request(index, prompt):
        return {
            "custom_id": str(index),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-5.4",
                "max_completion_tokens": 10,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            },
        }

    requests = [make_request(k, v) for k, v in user_prompts.items()]

    # Write the JSONL input file (one JSON object per line)
    input_file_path = annotation_dir / "input_file.jsonl"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    with open(input_file_path, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    # Upload the file to OpenAI
    print(f"Uploading {len(requests)} requests to OpenAI...")
    with open(input_file_path, "rb") as f:
        batch_file = client.files.create(file=f, purpose="batch")

    # Create the batch job
    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"Batch created. ID: {batch.id}")
    (annotation_dir / "batch_id.txt").write_text(batch.id)

    # Poll until the batch finishes
    print("Polling for completion (checks every 60s)...")
    while True:
        batch = client.batches.retrieve(batch.id)
        print(f"  Status: {batch.status}")
        if batch.status == "completed":
            break
        elif batch.status in ("failed", "expired", "cancelled"):
            raise RuntimeError(f"Batch ended with status: {batch.status}")
        time.sleep(60)

    # Download and save the output file (successful requests)
    print(f"Batch complete. Saving responses to {annotation_dir}")
    if batch.output_file_id:
        output_jsonl = client.files.content(batch.output_file_id).text
        (annotation_dir / "output_file.jsonl").write_text(output_jsonl)
        for line in output_jsonl.strip().splitlines():
            result = json.loads(line)
            index = int(result["custom_id"])
            response = result["response"]["body"]["choices"][0]["message"]["content"]
            save_prompt(annotation_dir, user_prompts[index], index)
            save_response(annotation_dir, response, index)
    else:
        print("Warning: no output file — all requests may have failed.")

    # Download and save the error file (failed requests), if any
    if batch.error_file_id:
        error_jsonl = client.files.content(batch.error_file_id).text
        (annotation_dir / "error_file.jsonl").write_text(error_jsonl)
        for line in error_jsonl.strip().splitlines():
            result = json.loads(line)
            index = int(result["custom_id"])
            save_error(annotation_dir, str(result.get("error", "unknown error")), index)
        print(f"Warning: some requests failed — see {annotation_dir}/error_file.jsonl")


########
# Main #
########

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Annotate a parsed CSV using an LLM API.")
    parser.add_argument("api", choices=["deepseek", "openai"],
                        help="Which API to use")
    parser.add_argument("dataset", choices=list(system_prompts.keys()),
                        help="Dataset name (must match annotate_prompts.py)")
    parser.add_argument("parsed_csv", type=Path,
                        help="Path to parsed scaled CSV file")
    parser.add_argument("annotation_dir", type=Path,
                        help="Directory to save prompts/responses into")
    parser.add_argument("--num", type=int, default=None,
                        help="Number of rows to annotate (default: all)")
    parser.add_argument("--start", type=int, default=0,
                        help="Row index to start from (default: 0)")
    parser.add_argument("--num_examples", type=int, default=0,
                        help="Number of few-shot examples to include (default: 0)")
    args = parser.parse_args()

    # Load data — only the "text" and "y" columns are read.
    # "y" is used only to sample few-shot examples; it is never sent to the LLM.
    # All x features remain in the CSV on disk and are not loaded here at all.
    data = pd.read_csv(args.parsed_csv, usecols=["text", "y"])
    data = data.iloc[args.start: args.start + args.num] if args.num else data.iloc[args.start:]
    data = data.reset_index(drop=True)

    # Sample few-shot examples from the loaded rows (if requested).
    # Samples one example per class first (stratified), then fills remaining
    # slots randomly. Labels are mapped to human-readable strings for the prompt,
    # but only these example rows — not the full y column — are shown to the LLM.
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

    # Build one prompt per row. Only "text" goes into the prompt — y is excluded.
    data["prompt"] = data["text"].apply(
        lambda text: make_user_prompt(args.dataset, text, examples if examples else None)
    )

    print(f"Annotating {len(data)} rows with {args.api} ({args.dataset})")
    print(f"Few-shot examples: {len(examples)}")
    print(f"Output directory: {args.annotation_dir}\n")

    args.annotation_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = system_prompts[args.dataset]
    user_prompts = {i: data["prompt"][i] for i in data.index}

    if args.api == "deepseek":
        annotate_deepseek(system_prompt, user_prompts, args.annotation_dir)
    elif args.api == "openai":
        annotate_openai(system_prompt, user_prompts, args.annotation_dir)

    print("\nDone. Run gather_responses.py to assemble the annotated CSV.")
