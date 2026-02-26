from pathlib import Path
import pandas as pd
from argparse import ArgumentParser
import os

###################
# Parse arguments #
###################

parser = ArgumentParser(description = "Parse the ParlaSent dataset")
parser.add_argument("original_path", type = Path)
parser.add_argument("parsed_path", type = Path)
args = parser.parse_args()

#########
# Paths #
#########

files = [args.original_path / Path(f) for f in os.listdir(args.original_path)]
data = [pd.read_json(file, lines = True) for file in files]
data = pd.concat(data, axis = 0, ignore_index = True)
print(data)

#####################
# Check for balance #
#####################

print("")

print("Number of samples from each country:")
for country in data["country"].unique():
    num_country = len(data[data["country"] == country])
    print(f" - {country}: {num_country}")

print("")

print("Number of sentiment labels:")
for label in data["label"].unique():
    num_label = len(data[data["label"] == label])
    print(f" - {label}: {num_label}")

#######################
# Balance the dataset #
#######################

positive = data[data["label"] == "Positive"]
negative = data[data["label"] == "Negative"]
negative = negative.sample(len(positive), ignore_index = True)
data = pd.concat([positive, negative], axis = 0, ignore_index = True)
print(data)

####################
# Extract features #
####################

data["text"] = data["sentence"]

# country of origin
countries = {country: i for i, country in enumerate(data["country"].unique())}
data["x1"] = data["country"].map(countries.get)

# party
parties = {party: i for i, party in enumerate(data["party"].unique())}
data["x2"] = data["party"].map(parties.get)

# binary gender of the MP
genders = {gender: i for i, gender in enumerate(data["gender"].unique())}
data["x3"] = data["gender"].map(genders.get)

# length in characters
data["x4"] = data["text"].map(len)

# binary outcome
data["y"] = data["label"].map(lambda x: x.upper())

#############
# Save data #
#############

data = data[["text", "x1", "x2", "x3", "x4", "y"]]
print("\nFinal data:")
print(data)

data.to_json(args.parsed_path)
print(f"\nSaved data to {args.parsed_path}")
