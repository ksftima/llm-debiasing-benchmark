import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Dataset paths (scaled versions)
datasets = {
    'CUAD': 'parsed_datasets/parsed_csv/parsed_cuad.csv',
    #'FOMC': 'parsed_datasets/parsed_scaled_datasets/parsed_fomc_scaled.csv',
    #'Misogynistic': 'parsed_datasets/parsed_scaled_datasets/parsed_misogynistic_scaled.csv',
    #'PubMedQA': 'parsed_datasets/parsed_scaled_datasets/parsed_pubmedqa_scaled.csv'
}

# Feature columns
feature_cols = ['x1', 'x2', 'x3', 'x4', 'x5']

print("Checking for collinearity in datasets:")

# Create output directory for heatmaps
output_dir = 'parsed_datasets/collinearity_heatmaps'
os.makedirs(output_dir, exist_ok=True)

for dataset_name, file_path in datasets.items():
    print(f"\nDataset: {dataset_name}")

    # Load dataset
    df = pd.read_csv(file_path)
    X = df[feature_cols]

    # Correlation Matrix
    print("\nCorrelation Matrix:")
    corr_matrix = X.corr()
    print(corr_matrix.round(3))

    # Save correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1)
    plt.title(f'{dataset_name} - Feature Correlation Matrix')
    plt.tight_layout()
    heatmap_path = f'{output_dir}/correlation_{dataset_name.lower()}.png'
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved heatmap: {heatmap_path}")

    # High correlations
    print("\nHigh correlations (|r| > 0.9):")
    high_corr_found = False
    for i in range(len(feature_cols)):
        for j in range(i+1, len(feature_cols)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.9:
                print(f"  {feature_cols[i]} <-> {feature_cols[j]}: {corr_val:.3f}")
                high_corr_found = True
    if not high_corr_found:
        print("  None found")

    # checking VIF
    print("\nVariance Inflation Factor (VIF):")
    vif_data = pd.DataFrame()
    vif_data["Feature"] = feature_cols
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(feature_cols))]
    vif_data = vif_data.sort_values('VIF', ascending=False)
    print(vif_data.to_string(index=False))

    max_vif = vif_data['VIF'].max()
    if max_vif < 5:
        print(f"Max VIF = {max_vif:.2f} - No serious collinearity")
    elif max_vif < 10:
        print(f"Max VIF = {max_vif:.2f} - Moderate collinearity")
    else:
        print(f"Max VIF = {max_vif:.2f} - High collinearity")
