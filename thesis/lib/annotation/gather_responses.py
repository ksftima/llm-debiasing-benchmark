"""
Step 2 of 2: Gather annotated responses and assemble the final CSV.

Reads the prompts/ and responses/ folders written by annotate_api.py,
parses the label out of each response, and joins back onto the original
parsed CSV to produce an annotated CSV with a y_hat column.

Rows where the response is missing or unparseable are dropped and reported.

Usage:
    python gather_responses.py <dataset> <parsed_csv> <annotation_dir> <output_csv>

Arguments:
    dataset         Dataset name: fomc | pubmedqa | cuad | misogynistic
    parsed_csv      Path to the original parsed scaled CSV (same one used in step 1)
    annotation_dir  Directory containing the prompts/ and responses/ folders
    output_csv      Where to write the final annotated CSV
"""

import os
import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from annotate_prompts import dataset_labels


def parse_response(text, labels):
    """
    Extract the integer label from a raw LLM response string.

    Counts how many times each label string appears in the response.
    Raises ValueError if:
      - more than one distinct label appears
      - the same label appears more than once
      - no label appears at all
    """
    counts = {i: text.count(label) for i, label in enumerate(labels)}
    n_distinct = sum(1 for n in counts.values() if n > 0)
    n_total = sum(counts.values())

    if n_distinct > 1:
        raise ValueError("Response contains more than one label")
    if n_total > 1:
        raise ValueError("Response contains multiple copies of the same label")
    if n_total == 0:
        raise ValueError("Response contains no label")

    return [k for k, v in counts.items() if v > 0][0]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Assemble annotated CSV from response files.")
    parser.add_argument("dataset", choices=list(dataset_labels.keys()),
                        help="Dataset name")
    parser.add_argument("parsed_csv", type=Path,
                        help="Path to the original parsed scaled CSV")
    parser.add_argument("annotation_dir", type=Path,
                        help="Directory containing prompts/ and responses/ folders")
    parser.add_argument("output_csv", type=Path,
                        help="Where to write the annotated output CSV")
    args = parser.parse_args()

    labels = dataset_labels[args.dataset]

    # Read all response files
    responses = {}
    response_dir = args.annotation_dir / "responses"
    for fname in os.listdir(response_dir):
        index = int(fname[len("response_"):-len(".txt")])
        responses[index] = (response_dir / fname).read_text(encoding="utf-8").strip()

    # Read matching prompt files
    prompts = {}
    for index in responses:
        prompt_file = args.annotation_dir / "prompts" / f"prompt_{index:05}.txt"
        try:
            prompts[index] = prompt_file.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            print(f"Warning: no prompt file for row {index}")

    # Parse labels from responses
    annotations = {}
    invalid = {}
    for index, response in responses.items():
        try:
            annotations[index] = parse_response(response, labels)
        except ValueError as e:
            print(f"Row {index:05} invalid — {e}")
            print(f"  Response: {repr(response)}")
            invalid[index] = response

    print(f"\nTotal responses:   {len(responses)}")
    print(f"Valid annotations: {len(annotations)}")
    print(f"Invalid responses: {len(invalid)}")

    # Load original CSV and join annotations back in
    data = pd.read_csv(args.parsed_csv)
    data = data.reset_index(drop=True)

    data["prompt"] = [prompts.get(i) for i in data.index]
    data["response"] = [responses.get(i) for i in data.index]
    data["y_hat"] = [annotations.get(i) for i in data.index]

    # Drop rows with no valid annotation
    n_before = len(data)
    data = data.dropna(subset=["y_hat"]).reset_index(drop=True)
    data["y_hat"] = data["y_hat"].astype(int)
    print(f"Rows in original CSV: {n_before}")
    print(f"Rows in output CSV:   {len(data)} (dropped {n_before - len(data)})")

    # Cohen's kappa: agreement between LLM (y_hat) and expert labels (y)
    kappa = cohen_kappa_score(data["y"], data["y_hat"])
    print(f"\nCohen's kappa (y vs y_hat): {kappa:.4f}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output_csv, index=False)
    print(f"\nSaved annotated CSV to: {args.output_csv}")
