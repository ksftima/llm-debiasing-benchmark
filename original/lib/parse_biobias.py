from pathlib import Path
import pandas as pd
from argparse import ArgumentParser

###################
# Parse arguments #
###################

parser = ArgumentParser(
    description = "Parse the Amazon dataset"
)
parser.add_argument("original_path", type = Path)
parser.add_argument("parsed_path", type = Path)
args = parser.parse_args()

#######################
# Balance the dataset #
#######################

data = pd.read_json(args.original_path)

labels = data[["profession", "gender"]]
data["label"] = labels.apply(lambda x: "_".join(x.astype(str)), axis = 1)
num_samples = 10000
num_labels = len(data["label"].unique())
samples_per_label = num_samples // num_labels
samples = []
for label in data["label"].unique():
    rows = data[data["label"] == label]
    sampled_rows = rows.sample(min(len(rows), samples_per_label))
    samples.append(sampled_rows)
samples = pd.concat(samples, axis = 0, ignore_index = True)
remaining = data[~data["hard_text"].isin(samples["hard_text"])]
remaining = remaining.sample(num_samples - len(samples))
data = pd.concat([samples, remaining], axis = 0, ignore_index = True)
data = data.drop("label", axis = 1)

#####################
# Check the samples #
#####################

print(samples)

print("Gender balance:")
for gender in samples["gender"].unique():
    num_gender = len(samples[samples["gender"] == gender])
    print(f" - {gender}: {num_gender}")

print("Profession balance:")
for profession in samples["profession"].unique():
    num_profession = len(samples[samples["profession"] == profession])
    print(f" - {profession}: {num_profession}")

####################
# Extract features #
####################

data["text"] = data["hard_text"]

# profession label
data["x1"] = data["profession"]

# number of characters
data["x2"] = data["text"].map(lambda x: len(x))

# number of words
data["x3"] = data["text"].map(lambda x: len(x.split(" ")))

# number of capital letters
data["x4"] = data["text"].map(lambda x: sum(1 for c in x if c.isupper()))

# binary outcome
data["y"] = data["gender"]

#############
# Save data #
#############

data = data[["text", "x1", "x2", "x3", "x4", "y"]]
print("\nFinal data:")
print(data)

data.to_json(args.parsed_path)
print(f"\nSaved data to {args.parsed_path}")
