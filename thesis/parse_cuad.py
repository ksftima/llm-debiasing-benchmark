import pandas as pd
import re
from pathlib import Path
from argparse import ArgumentParser
import sys

###################
# Parse arguments #
###################

parser = ArgumentParser(
    description = "Parse the CUAD License Grant dataset"
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

# Normalize accidental whitespace in column names (e.g., " index")
data.columns = data.columns.str.strip()

####################
# Extract features #
####################

# text length (character count)
data["x1"] = data["text"].map(lambda x: len(x))

# average word length
def avg_word_length(text):
    words = text.split()
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)
data["x2"] = data["text"].map(avg_word_length)

# punctuation count
def count_punctuation(text):
    return sum(1 for c in text if c in ".,;:!?()-\"'")
data["x3"] = data["text"].map(count_punctuation)

# legal jargon density proxy (count of common legal terms)
legal_terms = [
    "shall", "hereby", "pursuant", "thereof", "therein",
    "hereunder", "herein", "hereof", "provisions", "obligations",
    "effective", "breach", "assignment", "indemnification", "jurisdiction"
]

def legal_jargon_count(text):
    text_lower = text.lower()
    return sum(len(re.findall(r"\b" + re.escape(term) + r"\b", text_lower)) for term in legal_terms)

data["x4"] = data["text"].map(legal_jargon_count)

# licensing keyword presence/intensity
licensing_terms = [
    "license", "licensed", "licensee", "licensor",
    "grant", "granted", "rights", "right",
    "intellectual", "property", "exclusive",
    "royalty", "sublicense", "copyright",
    "trademark", "patent"
]

def licensing_keyword_count(text):
    text_lower = text.lower()
    return sum(len(re.findall(r"\b" + re.escape(term) + r"\b", text_lower)) for term in licensing_terms)

data["x5"] = data["text"].map(licensing_keyword_count)

# binary outcome
data["y"] = data["label"].astype(int)

#############
# Save data #
#############

data = data[["text", "x1", "x2", "x3", "x4", "x5", "y"]]
print("\nFinal data:")
print(data.head())

data.to_json(args.parsed_path)
print(f"\nSaved data to {args.parsed_path}")
