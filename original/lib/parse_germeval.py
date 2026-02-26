from pathlib import Path
import pandas as pd
from argparse import ArgumentParser

###################
# Parse arguments #
###################

parser = ArgumentParser(
    description = "Parse the Germeval18 dataset"
)
parser.add_argument("original_path", type = Path)
parser.add_argument("parsed_path", type = Path)
args = parser.parse_args()

#########
# Paths #
#########

train = pd.read_csv(args.original_path / "train.csv")
test = pd.read_csv(args.original_path / "test.csv")
data = pd.concat([train, test], axis = 0, ignore_index = True)
print(data)

#######################
# Balance the dataset #
#######################

other = data[data["binary"] == "OTHER"]
offense = data[data["binary"] == "OFFENSE"]
samples_per_class = min(len(other), len(offense))
print(samples_per_class)
other = other.sample(samples_per_class)
offense = offense.sample(samples_per_class)
data = pd.concat([other, offense], axis = 0, ignore_index = True)
print(data)

####################
# Extract features #
####################

# number of '@' characters
data["x1"] = data["text"].map(lambda x: sum(1 for c in x if c == '@'))

# number of characters
data["x2"] = data["text"].map(lambda x: len(x))

# number of words
data["x3"] = data["text"].map(lambda x: len(x.split(" ")))

# number of capital letters
data["x4"] = data["text"].map(lambda x: sum(1 for c in x if c.isupper()))

# binary outcome
data["y"] = data["binary"].map(lambda x: 0 if x == "OTHER" else 1)

#############
# Save data #
#############

data = data[["text", "x1", "x2", "x3", "x4", "y"]]
print("\nFinal data:")
print(data)

data.to_json(args.parsed_path)
print(f"\nSaved data to {args.parsed_path}")
