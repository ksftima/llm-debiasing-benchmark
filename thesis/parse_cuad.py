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

# x1: text length (character count)
data["x1"] = data["text"].map(lambda x: len(x))

# x2: average word length
def avg_word_length(text):
    words = text.split()
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)
data["x2"] = data["text"].map(avg_word_length)

# x3: capitalization ratio
def capitalization_ratio(text):
    words = text.split()
    if not words:
        return 0.0
    # Count words that start with capital letter (excluding first word)
    if len(words) == 1:
        return 0.0
    capitalized = sum(1 for w in words[1:] if w and w[0].isupper())
    return capitalized / (len(words) - 1)

data["x3"] = data["text"].map(capitalization_ratio)

# x4: legal jargon density proxy (count of common legal terms)
legal_terms = [
    "shall", "hereby", "pursuant", "thereof", "therein",
    "hereunder", "herein", "hereof", "provisions", "obligations",
    "effective", "breach", "assignment", "indemnification", "jurisdiction"
]

def legal_jargon_count(text):
    text_lower = text.lower()
    return sum(len(re.findall(r"\b" + re.escape(term) + r"\b", text_lower)) for term in legal_terms)

data["x4"] = data["text"].map(legal_jargon_count)

# x5: licensing keyword presence/intensity
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

data.to_csv(args.parsed_path.with_suffix(".csv"), index=False)
print(f"Saved data to {args.parsed_path.with_suffix('.csv')}")
