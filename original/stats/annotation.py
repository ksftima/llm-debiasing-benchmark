from argparse import ArgumentParser
from pathlib import Path

def list_subdir(path):
    return [file for file in path.iterdir() if file.is_dir()]

parser = ArgumentParser(description = "Stats about the annotations")
parser.add_argument("annotation_dir", type = Path)
args = parser.parse_args()

for dataset in list_subdir(args.annotation_dir):
    print(f"\nDataset: {dataset.parts[-1]}")
    for annotation in list_subdir(dataset):
        if annotation.parts[-1] == "original":
            pass
        else:
            try:
                response_dir = annotation / Path("responses")
                num_responses = len(list(response_dir.iterdir()))
            except FileNotFoundError:
                num_responses = 0
            print(f" - {annotation.parts[-1]}: {num_responses}")

print("")
