"""
Evaluate annotation quality for all annotated datasets.

Prints per-dataset summary including:
  - Accuracy
  - Cohen's kappa
  - Class distribution (y vs y_hat)
  - Label distribution shift (mean y vs mean y_hat)
  - Confusion matrix
  - Per-class precision, recall, F1

Output is written both to the terminal and to thesis/annotations/evaluation_results.txt.

Usage:
    python evaluate_annotations.py
"""

import pandas as pd
import sys
from pathlib import Path
from sklearn.metrics import (
    cohen_kappa_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)

OUTPUT_FILE = Path("thesis/annotations/evaluation_results.txt")

DATASETS = {
    "fomc (deepseek)":                Path("thesis/annotations/fomc/fomc_deepseek_annotated.csv"),
    "fomc (gpt4omini)":               Path("thesis/annotations/fomc/fomc_gpt4omini_annotated.csv"),
    "fomc (gpt54)":                   Path("thesis/annotations/fomc/fomc_gpt54_annotated.csv"),
    "fomc (fewshot-gpt54)":           Path("thesis/annotations/fomc/fomc_fewshot_gpt54_annotated.csv"),
    "pubmedqa (deepseek)":            Path("thesis/annotations/pubmedqa/pubmedqa_deepseek_annotated.csv"),
    "pubmedqa (gpt4omini)":           Path("thesis/annotations/pubmedqa/pubmedqa_gpt4omini_annotated.csv"),
    "pubmedqa (gpt54)":               Path("thesis/annotations/pubmedqa/pubmedqa_gpt54_annotated.csv"),
    "pubmedqa (fewshot-gpt54)":       Path("thesis/annotations/pubmedqa/pubmedqa_fewshot_gpt54_annotated.csv"),
    "cuad (deepseek)":                Path("thesis/annotations/cuad/cuad_deepseek_annotated.csv"),
    "cuad (gpt4omini)":               Path("thesis/annotations/cuad/cuad_gpt4omini_annotated.csv"),
    "cuad (gpt54)":                   Path("thesis/annotations/cuad/cuad_gpt54_annotated.csv"),
    "cuad (fewshot-gpt54)":           Path("thesis/annotations/cuad/cuad_fewshot_gpt54_annotated.csv"),
    "misogynistic (deepseek)":        Path("thesis/annotations/misogynistic/misogynistic_deepseek_annotated.csv"),
    "misogynistic (gpt54)":           Path("thesis/annotations/misogynistic/misogynistic_gpt54_annotated.csv"),
    "misogynistic (fewshot-gpt54)":   Path("thesis/annotations/misogynistic/misogynistic_fewshot_gpt54_annotated.csv"),
    "misogynistic (mistral)":         Path("thesis/annotations/misogynistic/misogynistic_mistral_annotated.csv"),
    "fomc (mistral)":                 Path("thesis/annotations/fomc/fomc_mistral_annotated.csv"),
    "pubmedqa (mistral)":             Path("thesis/annotations/pubmedqa/pubmedqa_mistral_annotated.csv"),
    "cuad (mistral)":                 Path("thesis/annotations/cuad/cuad_mistral_annotated.csv"),
    "fomc (llama)":                   Path("thesis/annotations/fomc/fomc_llama_annotated.csv"),
    "pubmedqa (llama)":               Path("thesis/annotations/pubmedqa/pubmedqa_llama_annotated.csv"),
    "cuad (llama)":                   Path("thesis/annotations/cuad/cuad_llama_annotated.csv"),
    "misogynistic (llama)":           Path("thesis/annotations/misogynistic/misogynistic_llama_annotated.csv"),
}

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

def p(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=output)

with open(OUTPUT_FILE, "w") as output:
    for dataset, path in DATASETS.items():
        if not path.exists():
            print(f"Skipping {dataset} — file not found: {path}")
            continue

        p("=" * 60)
        p(f"DATASET: {dataset.upper()}")
        p("=" * 60)

        data = pd.read_csv(path)
        y     = data["y"].astype(int)
        y_hat = data["y_hat"].astype(int)
        labels = sorted(y.unique())

        p(f"Rows:          {len(data)}")
        p(f"Accuracy:      {accuracy_score(y, y_hat):.4f}")
        p(f"Cohen's kappa: {cohen_kappa_score(y, y_hat):.4f}")

        p("\nClass distribution (count):")
        dist = pd.DataFrame({
            "y (expert)":  y.value_counts().sort_index(),
            "y_hat (LLM)": y_hat.value_counts().sort_index(),
        })
        dist["y % "]      = (dist["y (expert)"]  / len(y)     * 100).round(1)
        dist["y_hat %"]   = (dist["y_hat (LLM)"] / len(y_hat) * 100).round(1)
        p(dist.to_string())

        p(f"\nMean y (expert):  {y.mean():.4f}")
        p(f"Mean y_hat (LLM): {y_hat.mean():.4f}")
        p(f"Shift:            {y_hat.mean() - y.mean():+.4f}")

        p("\nConfusion matrix (rows=true, cols=predicted):")
        cm = confusion_matrix(y, y_hat, labels=labels)
        cm_df = pd.DataFrame(cm, index=[f"true={l}" for l in labels],
                                 columns=[f"pred={l}" for l in labels])
        p(cm_df.to_string())

        p("\nPer-class metrics:")
        p(classification_report(y, y_hat, digits=4))

print(f"\nResults saved to: {OUTPUT_FILE}")
