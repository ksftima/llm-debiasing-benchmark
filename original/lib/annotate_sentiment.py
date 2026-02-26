import pandas as pd
from transformers import pipeline
from pathlib import Path
from argparse import ArgumentParser

###################
# Parse arguments #
###################

parser = ArgumentParser(
    description = "Annotate the Amazon dataset"
)
parser.add_argument("parsed_path", type = Path)
parser.add_argument("annotated_path", type = Path)
args = parser.parse_args()

#############
# Load data #
#############

data = pd.read_json(args.parsed_path)
print(data)

######################
# Annotate with BERT #
######################

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model = "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    revision = "714eb0f",
)
# Truncate reviews to 512 characters to fit into DistilBERT
text_truncated = data["text"].map(lambda x: x[:512]).tolist()
sentiments = sentiment_pipeline(text_truncated)
data["y_hat"] = [1 if s["label"] == "POSITIVE" else 0 for s in sentiments]

##############
# Save data # 
##############

print("\nFinal data:")
print(data.head())

data.to_json(args.annotated_path)
print(f"\nSaved data to {args.annotated_path}")
