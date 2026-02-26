# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":

    datasets = ["amazon", "misinfo", "biobias", "germeval"]

    for dataset in datasets:
        print("")
        print(f"Dataset: {dataset}")

        print(" - loading the data")
        data = data = pd.read_json(f"/mimer/NOBACKUP/groups/ci-nlp-alvis/dsl-use/annotations/{dataset}/parsed.json")

        features = data[["x1","x2","x3","x4"]]
        print(f" - Feature means: {features.mean().to_numpy()}")
        print(f" - Feature std: {features.std().to_numpy()}")

        print("")
