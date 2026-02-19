import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

# Dataset paths
datasets = {
    'CUAD': 'parsed_datasets/parsed_csv/parsed_cuad.csv',
    'FOMC': 'parsed_datasets/parsed_csv/parsed_fomc.csv',
    'Misogynistic': 'parsed_datasets/parsed_csv/parsed_misogynistic.csv',
    'PubMedQA': 'parsed_datasets/parsed_csv/parsed_pubmedqa.csv'
}

# Feature columns to standardize
feature_cols = ['x1', 'x2', 'x3', 'x4', 'x5']

for dataset_name, file_path in datasets.items():
    print(f"Processing: {dataset_name}")

    # Load dataset
    df = pd.read_csv(file_path)
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

    # Save standardized dataset
    output_path = file_path.replace('.csv', '_scaled.csv')
    df_scaled.to_csv(output_path, index=False)
    print(f"\n Saved standardized dataset at {output_path}")

print("\nAll datasets have been standardized!")
