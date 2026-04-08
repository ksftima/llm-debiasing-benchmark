"""
Replace stale features in VUAMC annotated CSVs with the freshly scaled data.

Keeps y_hat, prompt, response from the annotated files.
Replaces text, x1-x5, y with values from parsed_scaled_datasets/vuamc.csv.
Joins on row index (both files are derived from the same raw data in the same order).

Usage:
    python update_vuamc_features.py
"""

from pathlib import Path
import pandas as pd

SCALED_CSV  = Path("thesis/datasets/parsed/parsed_scaled_datasets/vuamc.csv")
ANNOTATED_DIR = Path("thesis/datasets/annotated/vuamc")

scaled = pd.read_csv(SCALED_CSV)

for ann_path in sorted(ANNOTATED_DIR.glob("vuamc_*_annotated.csv")):
    ann = pd.read_csv(ann_path)

    if len(ann) != len(scaled):
        print(f"NOTE: {ann_path.name} has {len(ann)} rows — finding dropped rows by sequence alignment.")
        # Find which scaled rows are missing by walking both sequences
        ann_texts = ann["text"].tolist()
        scaled_texts = scaled["text"].tolist()
        kept = []
        j = 0
        for i, t in enumerate(scaled_texts):
            if j < len(ann_texts) and t == ann_texts[j]:
                kept.append(i)
                j += 1
        scaled_aligned = scaled.iloc[kept].reset_index(drop=True)
        print(f"  Aligned {len(scaled_aligned)} rows (dropped {len(scaled) - len(scaled_aligned)} unmatched)")
        merged = scaled_aligned.copy()
        merged["y_hat"]    = ann["y_hat"].values
        merged["prompt"]   = ann["prompt"].values
        merged["response"] = ann["response"].values
    else:
        merged = scaled.copy()
        merged["y_hat"]    = ann["y_hat"].values
        merged["prompt"]   = ann["prompt"].values
        merged["response"] = ann["response"].values

    merged.to_csv(ann_path, index=False)
    print(f"Updated {ann_path.name}")

print("Done.")
