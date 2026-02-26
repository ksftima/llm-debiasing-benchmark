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

#############
# Read data #
#############

data = pd.read_parquet(args.original_path)
print(data.head())
print(f"Total samples: {len(data)}")

sources = {s: len(data[data["source"] == s]) for s in data["source"].unique()}
sources = sorted(sources.items(), key = lambda x: x[1], reverse = True)
print("Top 10 sources")
for s in sources[:10]:
    print(f" - {s[0]}: {s[1]}")

# Keep only two balanced classes
thesun = data[data["source"] == "thesun"]
theguardianuk = data[data["source"] == "theguardianuk"]
n_class = min(5000, len(thesun), len(theguardianuk))
thesun = thesun.sample(n = n_class)
theguardianuk = theguardianuk.sample(n = n_class)
data = pd.concat([thesun, theguardianuk])
print(f"Filtered data samples: {len(data)}")

####################
# Extract features #
####################

data["text"] = data["content"]

# number of characters
data["x1"] = data["content"].map(lambda x: len(x))

# number of words
data["x2"] = data["content"].map(lambda x: len(x.split(" ")))

# number of capital letters
data["x3"] = data["content"].map(lambda x: sum(1 for c in x if c.isupper()))

# number of characters in title
data["x4"] = data["title"].map(lambda x: len(x))

# binary outcome
source_map = {"thesun": 0, "theguardianuk": 1}
data["y"] = data["source"].map(source_map)

#############
# Save data #
#############

data = data[["text", "x1", "x2", "x3", "x4", "y"]]
print("\nFinal data:")
print(data)

data.to_json(args.parsed_path)
print(f"\nSaved data to {args.parsed_path}")
