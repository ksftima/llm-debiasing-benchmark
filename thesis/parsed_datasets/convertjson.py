import pandas as pd
import json

data = pd.read_json("/Users/kesafatima/llm-debiasing-benchmark/thesis/parsed_datasets/parsed_fomc.json")
data.to_csv("/Users/kesafatima/llm-debiasing-benchmark/thesis/parsed_datasets/parsed_fomc_x.csv", index = False)
