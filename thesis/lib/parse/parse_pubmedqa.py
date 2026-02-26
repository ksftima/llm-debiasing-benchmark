import pandas as pd
import re
from pathlib import Path
from argparse import ArgumentParser
import sys

###################
# Parse arguments #
###################

parser = ArgumentParser(
    description = "Parse the PubMedQA dataset"
)
parser.add_argument("original_path", type = Path)
parser.add_argument("parsed_path", type = Path)
args = parser.parse_args()

###########################
# Parse data to DataFrame #
###########################

try:
    data = pd.read_csv(args.original_path)
except Exception as e:
    print(f"Failed to open the dataset at {args.original_path}: {e}")
    sys.exit(1)

data["text"] = data["question"] + " " + data["context"]

####################
# Extract features #
####################

# question length (character count)
data["x1"] = data["question"].map(lambda x: len(x))

# context length (character count)
data["x2"] = data["context"].map(lambda x: len(x))

# medical terminology density (words with >10 characters)
data["x3"] = data["context"].map(lambda x: sum(1 for word in x.split() if len(word) > 10))

# number of numerical values
def count_numbers(text):
    return len(re.findall(r"\d+\.?\d*%?", text))
data["x4"] = data["context"].map(count_numbers)

# hedging word count
hedging_words = ["may", "might", "could", "likely", "potential",
                 "unclear", "possibly", "suggested", "potentially",
                 "suggest", "suggests", "uncertain", "probably",
                 "seems", "appears", "unlikely"]
def count_hedging(text):
    text_lower = text.lower()
    return sum(len(re.findall(r"\b" + word + r"\b", text_lower)) for word in hedging_words)
data["x5"] = data["context"].map(count_hedging)

# label (yes=1, no=0, maybe=2)
label_map = {"yes": 1, "no": 0, "maybe": 2}
data["y"] = data["expert_label"].map(label_map)

#############
# Save data #
#############

data = data[["text", "x1", "x2", "x3", "x4", "x5", "y"]]
print("\nFinal data:")
print(data.head())

data.to_json(args.parsed_path)
print(f"\nSaved data to {args.parsed_path}")