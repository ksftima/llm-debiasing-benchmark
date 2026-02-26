from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import DistilBertTokenizer, DistilBertModel
import numpy as np
import pandas as pd
import torch
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

#################
# Test features #
#################

X = data[["x1","x2","x3","x4"]].to_numpy()
Y = data["y"].to_numpy()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

logreg = LogisticRegression(max_iter = 1000)
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
print(f"Accuracy: {accuracy_score(Y_test, Y_pred)}")

############
# Annotate #
############

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device {device}")

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)

def embed(texts, batch_size = 32):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors = "pt",
            padding = True,
            truncation = True,
            max_length = 512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state[:,0,:].cpu().numpy() # Use CLS token
        embeddings.extend(batch_embeddings)
    return embeddings

embeddings = embed(list(data["text"]))
X = np.array(embeddings)
print(f"Embedding dimensions: {X.shape}")
Y = data["y"].to_numpy()
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X, Y)
data["y_hat"] = logreg.predict(X)

#############
# Save data #
#############

print("\nFinal data:")
print(data)

data.to_json(args.annotated_path)
print(f"\nSaved data to {args.annotated_path}")
