"""
Cancel a stalled OpenAI batch job, recover completed responses,
and annotate remaining rows via real-time calls.

Usage:
    python recover_batch.py <dataset> <parsed_csv> <annotation_dir>

Example:
    python recover_batch.py fomc \
        thesis/datasets/parsed/parsed_scaled_datasets/fomc.csv \
        thesis/datasets/annotated/fomc/openai/gpt54/
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import batched

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent / "thesis/lib/annotation"))
from annotate_prompts import dataset_labels, system_prompts, make_user_prompt

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


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


def cancel_and_recover(annotation_dir):
    batch_id = (annotation_dir / "batch_id.txt").read_text().strip()
    print(f"Cancelling batch {batch_id}...")
    client.batches.cancel(batch_id)

    while True:
        batch = client.batches.retrieve(batch_id)
        print(f"  Status: {batch.status} | completed: {batch.request_counts.completed}")
        if batch.status in ("cancelled", "expired", "failed"):
            break
        time.sleep(5)

    recovered = {}
    if batch.output_file_id:
        print("Downloading partial output...")
        output_jsonl = client.files.content(batch.output_file_id).text
        (annotation_dir / "output_file.jsonl").write_text(output_jsonl)
        for line in output_jsonl.strip().splitlines():
            result = json.loads(line)
            index = int(result["custom_id"])
            response = result["response"]["body"]["choices"][0]["message"]["content"]
            recovered[index] = response
        print(f"Recovered {len(recovered)} responses from batch output.")
    else:
        print("No output file available.")

    return recovered


def annotate_realtime(missing_indices, user_prompts, system_prompt, annotation_dir):
    print(f"\nRunning real-time calls for {len(missing_indices)} missing rows...")

    def call_one(index):
        response = client.chat.completions.create(
            model="gpt-5.4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompts[index]},
            ],
            max_completion_tokens=10,
        )
        return index, response.choices[0].message.content

    done = 0
    for batch_indices in batched(missing_indices, 50):
        batch_indices = list(batch_indices)
        with ThreadPoolExecutor(max_workers=len(batch_indices)) as executor:
            futures = {executor.submit(call_one, i): i for i in batch_indices}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    idx, response = future.result()
                    save_response(annotation_dir, response, idx)
                    done += 1
                except Exception as e:
                    save_error(annotation_dir, str(e), index)
                    print(f"  Row {index:05} failed: {e}")
        print(f"  {done}/{len(missing_indices)} done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("parsed_csv", type=Path)
    parser.add_argument("annotation_dir", type=Path)
    args = parser.parse_args()

    annotation_dir = args.annotation_dir

    # Cancel batch and recover completed responses
    recovered = cancel_and_recover(annotation_dir)

    # Save recovered responses to disk
    responses_dir = annotation_dir / "responses"
    already_on_disk = {int(f.stem[len("response_"):]) for f in responses_dir.glob("response_*.txt")} if responses_dir.exists() else set()

    for index, response in recovered.items():
        if index not in already_on_disk:
            save_response(annotation_dir, response, index)

    all_saved = {int(f.stem[len("response_"):]) for f in (annotation_dir / "responses").glob("response_*.txt")}
    print(f"Total responses on disk: {len(all_saved)}")

    # Load data and build prompts
    data = pd.read_csv(args.parsed_csv, usecols=["text", "y"]).reset_index(drop=True)
    system_prompt = system_prompts[args.dataset]
    user_prompts = {i: make_user_prompt(args.dataset, row["text"], None) for i, row in data.iterrows()}

    # Save prompts for recovered rows
    for index in recovered:
        prompt_file = annotation_dir / "prompts" / f"prompt_{index:05}.txt"
        if not prompt_file.exists():
            save_prompt(annotation_dir, user_prompts[index], index)

    # Find missing rows
    missing = sorted(set(data.index) - all_saved)
    print(f"Missing rows: {len(missing)}")

    if missing:
        annotate_realtime(missing, user_prompts, system_prompt, annotation_dir)
        for index in missing:
            save_prompt(annotation_dir, user_prompts[index], index)

    print("\nDone. Run gather_responses.py to assemble the annotated CSV.")
