import pandas as pd
import re
from pathlib import Path
from argparse import ArgumentParser
import sys

###################
# Parse arguments #
###################

parser = ArgumentParser(
    description="Parse the VUAMC metaphor dataset"
)
parser.add_argument("original_path", type=Path)
parser.add_argument("parsed_path", type=Path)
args = parser.parse_args()

###########################
# Parse data to DataFrame #
###########################

try:
    data = pd.read_csv(args.original_path)
except Exception as e:
    print(f"Failed to open the dataset at {args.original_path}: {e}")
    sys.exit(1)

# Drop rows with missing sentence
n_before = len(data)
data = data.dropna(subset=["sentence"]).reset_index(drop=True)
n_dropped = n_before - len(data)
if n_dropped > 0:
    print(f"Dropped {n_dropped} rows with missing sentence text.")

####################
# Extract features #
####################

# text column (rename from 'sentence')
data["text"] = data["sentence"].astype(str)

# x1: word count
data["x1"] = data["text"].map(lambda x: len(x.split()))

# x2: average word length
def avg_word_length(text):
    words = text.split()
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)

data["x2"] = data["text"].map(avg_word_length)

# x3: punctuation density (! ? , . ; : per character)
def punctuation_density(text):
    if not text:
        return 0.0
    punct = sum(1 for ch in text if ch in "!?,.:;")
    return punct / len(text)

data["x3"] = data["text"].map(punctuation_density)

# x4: copula count (is, was, are, were, be, been, being)
# Metaphors often use copular constructions ("X is Y")
copula_words = {"is", "was", "are", "were", "be", "been", "being"}
def copula_count(text):
    words = re.findall(r"\b\w+\b", text.lower())
    return sum(1 for w in words if w in copula_words)

data["x4"] = data["text"].map(copula_count)

# x5: capital letter ratio (uppercase letters / total letters)
def cap_ratio(text):
    letters = re.findall(r"[A-Za-z]", text)
    if not letters:
        return 0.0
    caps = sum(1 for ch in letters if ch.isupper())
    return caps / len(letters)

data["x5"] = data["text"].map(cap_ratio)

# label (binary metaphor target)
data["y"] = data["has_metaphor"].astype(int)

#############
# Save data #
#############

data = data[["text", "x1", "x2", "x3", "x4", "x5", "y"]]
print("\nFinal data:")
print(data.head())
print(f"\nShape: {data.shape}")
print(f"\nFeature statistics:")
print(data[["x1", "x2", "x3", "x4", "x5"]].describe().round(4))
print(f"\nLabel distribution:\n{data['y'].value_counts()}")

data.to_json(args.parsed_path)
print(f"\nSaved data to {args.parsed_path}")
