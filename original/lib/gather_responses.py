from pathlib import Path
import pandas as pd 
import argparse
import os
from sklearn.metrics import cohen_kappa_score
from annotate_prompts import dataset_labels 

def parse_annotation(text, labels):
    # Assumes that labels are not subsets of each other...
    counts = {i: text.count(label) for i, label in enumerate(labels)}
    if 1 < sum(1 for n in counts.values() if n > 0):
        raise ValueError("Response contains more than one label")
    elif 1 < sum(counts.values()):
        raise ValueError("Response contains multiple copies of a label")
    elif 0 == sum(counts.values()):
        raise ValueError("Response didn't contain any label")
    else:
        return [k for k, v in counts.items() if v > 0][0]

def parse_biobias(text):
    male_count = text.count("MALE")
    female_count = text.count("FEMALE")
    if 0 < female_count and female_count < male_count:
        raise ValueError("Response contains both labels")
    elif 1 < male_count:
        raise ValueError("Response contains multiple copies of a label")
    elif 0 == female_count + male_count:
        raise ValueError("Response contains no labels")
    elif 1 == female_count:
        return 1
    elif 1 == male_count:
        return 0

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices = list(dataset_labels.keys()))
    parser.add_argument('parsed_path', type = Path)
    parser.add_argument('annotation_dir', type = Path)
    parser.add_argument('annotated_path', type = Path)
    args = parser.parse_args()

    responses = {}
    response_path = args.annotation_dir / Path("responses")
    for file in os.listdir(response_path):
        index = int(file[len("response_"):-len(".txt")])
        with open(response_path / file, "r", encoding = "utf-8") as f:
            responses[index] = f.read().strip()

    prompts = {}
    for index in responses.keys():
        try:
            prompt_file = args.annotation_dir / Path(f"prompts/prompt_{index:05}.txt")
            with open(prompt_file, "r", encoding = "utf-8") as f:
                prompts[index] = f.read().strip()
        except Exception as e:
            print(f"Couldn't find prompt file for index {index}: {e}")

    invalid_responses = {}
    annotations = {}
    labels = dataset_labels[args.dataset]
    for index, response in responses.items():
        try:
            if args.dataset == "biobias":
                annotations[index] = parse_biobias(response)
            else:
                annotations[index] = parse_annotation(response, labels)
        except ValueError as error:
            print(f"Value error at index {index}: {error}")
            print("- Response:", response)
            invalid_responses[index] = response
        except Exception as exception:
            print(f"Some exception occurred when parsing responses: {exception}")

    data = pd.read_json(args.parsed_path)

    print("Number of rows in parsed dataframe:", len(data))
    print("Number of prompts:", len(prompts))
    print("Number of responses:", len(responses))
    print("Number of annotations:", len(annotations))
    print("Number of invalid responses:", len(invalid_responses))

    data = data.reset_index(drop = True)
    data["prompt"] = [prompts.get(i, None) for i in data.index]
    data["response"] = [responses.get(i, None) for i in data.index]
    data["y_hat"] = [annotations.get(i, None) for i in data.index]
    data = data.dropna(subset = ["y_hat"])
    data = data[["x1", "x2", "x3", "x4", "y", "y_hat", "text", "prompt", "response"]]
    print(data)

    print("\nCohen kappa:", cohen_kappa_score(data["y"], data["y_hat"]))

    data.to_json(args.annotated_path)
    print(f"\nSaved data to {args.annotated_path}")
