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

# contract text (supports both old and new CUAD schemas)
if "text" in data.columns:
    data["text"] = data["text"].astype(str)
elif "contract_text" in data.columns:
    data["text"] = data["contract_text"].astype(str)
else:
    print("Dataset must contain either 'text' or 'contract_text' column.")
    sys.exit(1)

# contract length (character count)
data["x1"] = data["text"].map(lambda x: len(x))

# sentence length (average character count per sentence)
def avg_sentence_length(text):
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    if not sentences:
        return 0.0
    return sum(len(s) for s in sentences) / len(sentences)

data["x2"] = data["text"].map(avg_sentence_length)

# number of sections/articles (Section, Article, numbered clauses)
def section_article_count(text):
    count_section = len(re.findall(r"\bsection\b", text, flags=re.IGNORECASE))
    count_article = len(re.findall(r"\barticle\b", text, flags=re.IGNORECASE))
    # Count only top-level numbered clauses like "1.", "2.", "3." (not "1.2", "1.3", etc.)
    count_numbered = len(re.findall(r"(?m)^\s*\d+\.(?!\d)", text))
    return count_section + count_article + count_numbered

data["x3"] = data["text"].map(section_article_count)

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
