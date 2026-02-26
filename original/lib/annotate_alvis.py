import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from annotate_prompts import system_prompts, dataset_labels, make_user_prompt
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
from itertools import batched
import os

###########
# Options #
###########

parser = ArgumentParser()
parser.add_argument('dataset', choices = list(system_prompts.keys()))
parser.add_argument('parsed_path', type = Path)
parser.add_argument('annotation_dir', type = Path)
parser.add_argument('--num', type = int, default = 5)
parser.add_argument('--start', type = int, default = 0)
parser.add_argument('--model', type = str, default = "microsoft/phi-4")
parser.add_argument('--batchsize', type = int, default = 8)
parser.add_argument('--num_examples', type = int, default = 0)
args = parser.parse_args()
print(f"Running {args.num} DeepSeek API requests")
print(f"Start: {args.start}")

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"
print("Using device:", device)

#################
# Load the data #
#################

data = pd.read_json(args.parsed_path)
data = data[args.start:args.start + args.num]
data = data.reset_index(drop = True)
print(data)

###################
# Select examples #
###################

examples = data.sample(min(args.num, args.num_examples))
data = data.loc[~data.index.isin(examples.index)]
labels = examples["y"].apply(lambda i: dataset_labels[args.dataset][i])
examples = list(zip(examples["text"], labels))

############
# Annotate #
############

print(f"Loading tokenizer for {args.model}")
tokenizer = AutoTokenizer.from_pretrained(args.model)

print(f"Loading model for {args.model}")
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    device_map = device,
    torch_dtype = torch.float16,
)

generator = pipeline(
    "text-generation",
    model = model,
    tokenizer = tokenizer,
    max_new_tokens = 512,
    num_return_sequences = 1,
)

def make_chat(text):
    return [
        {
            "role": "system",
            "content":system_prompts[args.dataset],
        },
        {
            "role": "user",
            "content": make_user_prompt(args.dataset, text, examples),
        },
    ]

prompt_dir = args.annotation_dir / Path("prompts")
response_dir = args.annotation_dir / Path("responses")

os.makedirs(prompt_dir, exist_ok = True)
os.makedirs(response_dir, exist_ok = True)

for batch in batched(data.index, args.batchsize):
    texts = [data["text"][i] for i in batch]
    chats = [make_chat(text) for text in texts]
    outputs = generator(chats)

    for i, output in zip(batch, outputs):

        prompt_file = prompt_dir / Path(f"prompt_{args.start + i:04}.txt")
        print(prompt_file)
        with open(prompt_file, "w", encoding="utf-8") as file:
            file.write(output[0]["generated_text"][-2]["content"])

        response_file = response_dir / Path(f"response_{args.start + i:04}.txt")
        print(response_file)
        with open(response_file, "w", encoding="utf-8") as file:
            file.write(output[0]["generated_text"][-1]["content"])
