"""
Merge PPI++ summary CSVs (from Vera) into the originals in summaries/original/.

After running the PPI++ jobs on Vera and SCP-ing the summaries-ppipp/ folder locally,
run this script to append ppipp rows to each original CSV.

Usage:
    python3 merge_ppipp_results.py <ppipp_summaries_dir> <originals_dir>

Example:
    python3 merge_ppipp_results.py \
        thesis/results/summaries-ppipp \
        thesis/results/summaries/original

The script matches files by (dataset, llm, phase) and appends ppipp rows.
Original files are updated in-place (original content preserved, ppipp rows appended).
"""

import sys
import pandas as pd
from pathlib import Path


PHASE_SUFFIX_MAP = {
    "full_logistic": "full_logistic",
    "low_variance":  "low_variance",
    "high_variance": "high_variance",
}


def merge_one(ppipp_csv: Path, originals_dir: Path) -> None:
    """Append ppipp rows from ppipp_csv into the matching original CSV."""
    ppipp_df = pd.read_csv(ppipp_csv)

    if ppipp_df.empty:
        print(f"  SKIP (empty): {ppipp_csv.name}")
        return

    dataset = ppipp_df["dataset"].iloc[0]
    llm     = ppipp_df["llm"].iloc[0]
    phase   = ppipp_df["phase"].iloc[0]
    phase_suffix = PHASE_SUFFIX_MAP.get(phase, phase)

    original_path = originals_dir / f"{dataset}_{llm}_{phase_suffix}.csv"
    if not original_path.exists():
        print(f"  SKIP (original not found): {original_path.name}")
        return

    original_df = pd.read_csv(original_path)

    # Drop any existing ppipp rows so we can re-merge cleanly
    if "ppipp" in original_df["method"].values:
        original_df = original_df[original_df["method"] != "ppipp"]

    merged = pd.concat([original_df, ppipp_df], ignore_index=True)
    merged.to_csv(original_path, index=False)
    print(f"  Merged {len(ppipp_df)} ppipp rows → {original_path.name}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 merge_ppipp_results.py <ppipp_dir> <originals_dir>")
        sys.exit(1)

    ppipp_dir    = Path(sys.argv[1])
    originals_dir = Path(sys.argv[2])

    if not ppipp_dir.exists():
        print(f"Error: ppipp dir not found: {ppipp_dir}")
        sys.exit(1)
    if not originals_dir.exists():
        print(f"Error: originals dir not found: {originals_dir}")
        sys.exit(1)

    ppipp_csvs = sorted(ppipp_dir.glob("*_ppipp.csv"))
    if not ppipp_csvs:
        print(f"No *_ppipp.csv files found in {ppipp_dir}")
        sys.exit(1)

    print(f"Found {len(ppipp_csvs)} ppipp CSV(s) to merge.\n")

    for csv in ppipp_csvs:
        merge_one(csv, originals_dir)

    print("\nDone.")
