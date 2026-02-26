import pandas as pd
from pathlib import Path
from gather_responses import parse_annotation, parse_biobias
from annotate_prompts import dataset_labels

ann_dir = Path("results/annotations/amazon/")

data = pd.read_json(ann_dir / Path("parsed.json"))
print(data)

responses = pd.read_json(ann_dir / Path("openai/output_file.jsonl"), lines = True)
data = data[list(responses["custom_id"])]
print(data)

labels = dataset_labels["amazon"]
data["y_hat"] = responses["response"].map(lambda x: x["body"]["choices"][0]["message"]["content"])
data["y_hat"] = data["y_hat"].map(lambda x: parse_annotation(x, labels))
print(data)
