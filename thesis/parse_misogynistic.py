import pandas as pd
import re
from pathlib import Path
from argparse import ArgumentParser
import sys

###################
# Parse arguments #
###################

parser = ArgumentParser(
    description = "Parse the Misogynistic dataset"
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

####################
# Extract features #
####################

# text column
data["text"] = data["text"].astype(str)

# text length (character count)
data["x1"] = data["text"].map(lambda x: len(x))

# capital letter ratio (caps / total letters)
def cap_ratio(text):
    letters = re.findall(r"[A-Za-z]", text)
    if not letters:
        return 0.0
    caps = sum(1 for ch in letters if ch.isupper())
    return caps / len(letters)

data["x2"] = data["text"].map(cap_ratio)

# profanity/slur count
problematic_words = [
    "bitch", "slut", "whore", "cunt", "feminazi",
    "hoe", "skank", "hag", "thot", "gold digger"
]

def count_problematic(text):
    text_lower = text.lower()
    return sum(len(re.findall(r"\b" + re.escape(word) + r"\b", text_lower)) for word in problematic_words)

data["x3"] = data["text"].map(count_problematic)

# pronoun/gendered word count
gendered_words = ["she", "her", "woman", "women", "female"]

def count_gendered(text):
    text_lower = text.lower()
    return sum(len(re.findall(r"\b" + re.escape(word) + r"\b", text_lower)) for word in gendered_words)

data["x4"] = data["text"].map(count_gendered)

# punctuation intensity: exclamation + question marks
data["x5"] = data["text"].map(lambda x: x.count("!") + x.count("?"))

# label (binary misogynistic target)
data["y"] = data["label"].astype(int)

#############
# Save data #
#############

data = data[["text", "x1", "x2", "x3", "x4", "x5", "y"]]
print("\nFinal data:")
print(data.head())

data.to_json(args.parsed_path)
print(f"\nSaved data to {args.parsed_path}")
