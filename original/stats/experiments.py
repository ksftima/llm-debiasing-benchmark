from argparse import ArgumentParser
from pathlib import Path

def list_subdir(path):
    return [file for file in path.iterdir() if file.is_dir()]

parser = ArgumentParser(description = "Stats about the experiments")
parser.add_argument("experiment_dir", type = Path)
args = parser.parse_args()

experiment = args.experiment_dir / Path("vary-num-expert")
print(f"\n{experiment.parts[-1]}:")
for dataset in list_subdir(experiment):
    print(f" > {dataset.parts[-1]}")
    data_dir = dataset / Path("data")
    if data_dir.exists():
        for annotation in list_subdir(data_dir):
            repetitions = len(list(annotation.iterdir()))
            print(f"   - {annotation.parts[-1]}: {repetitions}")

print("")

experiment = args.experiment_dir / Path("vary-num-total")
print(f"\n{experiment.parts[-1]}:")
for dataset in list_subdir(experiment):
    print(f" > {dataset.parts[-1]}")
    data_dir = dataset / Path("data")
    if data_dir.exists():
        for annotation in list_subdir(data_dir):
            print(f"   - {annotation.parts[-1]}")
            for n in list_subdir(annotation):
                repetitions = len(list(n.iterdir()))
                print(f"     - {n.parts[-1]}: {repetitions}")
