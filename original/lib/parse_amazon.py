import pandas as pd
import re
from pathlib import Path
from argparse import ArgumentParser
import sys

###################
# Parse arguments #
###################

parser = ArgumentParser(
    description = "Parse the Amazon dataset"
)
parser.add_argument("original_path", type = Path)
parser.add_argument("parsed_path", type = Path)
args = parser.parse_args()

###########################
# Parse data to DataFrame #
###########################

try:
    with open(args.original_path, "r") as file:
        reviews = file.read().splitlines()
except Exception as e:
    print(f"Failed to open the dataset at {args.original_path}: {e}")
    sys.exit(1)

data = {k: [] for k in ["topic", "sentiment", "text"]}
for review in reviews:
    words = review.split(" ")
    data["topic"].append(words[0])
    data["sentiment"].append(1 if words[1] == "pos" else 0)
    data["text"].append(" ".join(words[3:]))

data = pd.DataFrame(data)
original_labels = list(data.columns.values)

####################
# Extract features #
####################

# topic
topics = pd.unique(data["topic"])
topicNum = dict(zip(topics, range(len(topics) + 1)))
data["x1"] = data["topic"].map(lambda x: topicNum[x])

# number of characters
data["x2"] = data["text"].map(lambda x: len(x))

# number of words
data["x3"] = data["text"].map(lambda x: len(x.split(" ")))

# number of repetitions of the word i
def repetitions_i(text):
    return sum(1 for _ in re.finditer(r"\b%s\b" % re.escape("i"), text))
data["x4"] = data["text"].map(lambda x: repetitions_i(x))

# binary outcome
data["y"] = data["sentiment"]

##############
# Save data # 
##############

data = data.drop(columns = ["topic", "sentiment"])
print("\nFinal data:")
print(data.head())

data.to_json(args.parsed_path)
print(f"\nSaved data to {args.parsed_path}")
