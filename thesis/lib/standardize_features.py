import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

# Dataset paths: input from parsed_csv/, output to parsed_scaled_datasets/
datasets = {
    'CUAD': ('thesis/datasets/parsed/parsed_csv/parsed_cuad.csv', 'thesis/datasets/parsed/parsed_scaled_datasets/cuad.csv'),
    'FOMC': ('thesis/datasets/parsed/parsed_csv/parsed_fomc.csv', 'thesis/datasets/parsed/parsed_scaled_datasets/fomc.csv'),
    'Misogynistic': ('thesis/datasets/parsed/parsed_csv/parsed_misogynistic.csv', 'thesis/datasets/parsed/parsed_scaled_datasets/misogynistic.csv'),
    'PubMedQA': ('thesis/datasets/parsed/parsed_csv/parsed_pubmedqa.csv', 'thesis/datasets/parsed/parsed_scaled_datasets/pubmedqa.csv'),
    'VUAMC': ('thesis/datasets/parsed/parsed_csv/parsed_vuamc.csv', 'thesis/datasets/parsed/parsed_scaled_datasets/vuamc.csv'),
}

# Feature columns to standardize
feature_cols = ['x1', 'x2', 'x3', 'x4', 'x5']

for dataset_name, (input_path, output_path) in datasets.items():
    print(f"Processing: {dataset_name}")

    # Load dataset
    df = pd.read_csv(input_path)
    print(f"Original shape: {df.shape}")

    # Display original statistics
    print("\nOriginal feature statistics:")
    print(df[feature_cols].describe().round(3))

    # Initialize StandardScaler
    scaler = StandardScaler()

    # Fit and transform the features
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Display scaled statistics
    print("\nScaled feature statistics:")
    print(df_scaled[feature_cols].describe().round(3))

    df_scaled.to_csv(output_path, index=False)
    print(f"\n Saved standardized dataset at {output_path}")

print("\nAll datasets have been standardized!")
