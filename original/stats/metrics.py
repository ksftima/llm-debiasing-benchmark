import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
from itertools import combinations

if __name__ == "__main__":

    datasets = ["amazon", "misinfo", "biobias", "germeval"]
    annotations = ["bert", "deepseek", "phi4", "claude"]

    annotation_dir = Path("/mimer/NOBACKUP/groups/ci-nlp-alvis/dsl-use/annotations")

    print("")
    print("Scores for gold-standard logistic regression")
    for dataset in datasets:
        print(f" - {dataset}")
        data = data = pd.read_json(annotation_dir / dataset / "parsed.json")
        features = ["x1","x2","x3","x4"]
        train, test = train_test_split(data, test_size = 0.2)
        logreg = LogisticRegression()
        logreg.fit(train[features], train["y"])
        preds = logreg.predict(test[features])
        print(f"     - accuracy: {accuracy_score(preds, test['y'])}")
        print(f"     - f1 score: {f1_score(preds, test['y'])}")

    print("")
    print("Agreement scores between predicted and gold-standard labels")
    for dataset in datasets:
        print(f" - {dataset}")
        for annotation in annotations:
            data_path = annotation_dir / dataset / f"annotated_{annotation}.json"
            data = data = pd.read_json(data_path)
            agreement = accuracy_score(data["y"], data["y_hat"])
            print(f"     - {annotation} accuracy: {agreement}")

    print("")
    print("Pearson r^2 correlation between features")
    feature_pairs = list(combinations(["x1","x2","x3","x4"], 2))
    to_name = lambda x, y: f"({x},{y})"
    correlations = {to_name(x,y): [] for x, y in feature_pairs}
    correlations["name"] = []
    for dataset in datasets:
        for annotation in annotations:
            correlations["name"].append(f"{dataset}/{annotation}")
            filepath = annotation_dir / dataset / f"annotated_{annotation}.json"
            data = data = pd.read_json(filepath)
            for x, y in feature_pairs:
                if x == y:
                    continue
                correlation = data[x].corr(data[y], method="pearson")
                correlations[to_name(x, y)].append(correlation**2)
    correlations = pd.DataFrame(correlations)
    print(correlations)
    correlations.to_csv("correlations.csv")

    print("")
