from datasets import load_dataset
from pathlib import Path
import sys

data_path = Path(sys.argv[1])

data = load_dataset("LabHC/bias_in_bios")
data = data["train"].to_pandas()

data.to_json(data_path)
print(f"\nSaved data to {data_path}")
