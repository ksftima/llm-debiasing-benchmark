import pandas as pd
import re
from pathlib import Path
from argparse import ArgumentParser
import sys

###################
# Parse arguments #
###################

parser = ArgumentParser(
    description = "Parse the FOMC dataset"
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

data["text"] = data["sentence"]

####################
# Extract features #
####################

# year (numeric)
data["x1"] = data["year"]

# sentence length (character count)
data["x2"] = data["sentence"].map(lambda x: len(x))

# positive economic keyword count
positive_words = ["growth", "increase", "strong", "higher", "rising",
                  "raise", "above", "elevated", "upward", "pressures"]
def count_positive(text):
    text_lower = text.lower()
    return sum(len(re.findall(r"\b" + word + r"\b", text_lower)) for word in positive_words)
data["x3"] = data["sentence"].map(count_positive)

# negative economic keyword count
negative_words = ["lower", "decline", "slow", "slowing", "easing",
                  "downward", "below", "weak", "weaken", "cut",
                  "downturn", "reduce"]
def count_negative(text):
    text_lower = text.lower()
    return sum(len(re.findall(r"\b" + word + r"\b", text_lower)) for word in negative_words)
data["x4"] = data["sentence"].map(count_negative)

# uncertainty markers
uncertainty_words = ["could", "expected", "likely", "outlook", "may",
                     "anticipated", "moderate", "uncertainty", "projected",
                     "gradually", "gradual", "expects", "uncertain"]
def count_uncertainty(text):
    text_lower = text.lower()
    return sum(len(re.findall(r"\b" + word + r"\b", text_lower)) for word in uncertainty_words)
data["x5"] = data["sentence"].map(count_uncertainty)

# label (dovish=0, neutral=1, hawkish=2)
data["y"] = data["label"]

#############
# Save data #
#############

data = data[["text", "x1", "x2", "x3", "x4", "x5", "y"]]
print("\nFinal data:")
print(data.head())

data.to_json(args.parsed_path)
print(f"\nSaved data to {args.parsed_path}")
