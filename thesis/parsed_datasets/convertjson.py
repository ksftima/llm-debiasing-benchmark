import pandas as pd
import json

data = pd.read_json("/Users/kesafatima/llm-debiasing-benchmark/thesis/parsed_datasets/parsed_misogynistic.json")
data.to_csv("/Users/kesafatima/llm-debiasing-benchmark/thesis/parsed_datasets/parsed_misogynistic_x.csv", index = False)
