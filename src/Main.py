import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load data
file_path = r'C:\Users\spac-23\Documents\w10\vgsales.csv'
data = pd.read_csv(file_path, header=0)

# Select numerical columns
numerical_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
X = data[numerical_cols].dropna()  # Drop rows with missing values

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform SVD
U, S, Vt = np.linalg.svd(X_scaled, full_matrices=False)
explained_variance = (S**2) / (X_scaled.shape[0] - 1)
explained_variance_ratio = explained_variance / explained_variance.sum()

# PCA Loadings
loadings = Vt.T  # Rows correspond to features, columns to PCs

# Correlation Circle (PC1 vs PC2)
plt.figure(figsize=(8, 8))
plt.axhline(0, color='gray', linestyle='--', linewidth=0.7)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.7)

for i, feature in enumerate(numerical_cols):
    plt.arrow(0, 0, loadings[i, 0], loadings[i, 1],
              head_width=0.05, head_length=0.05, color='b', alpha=0.7)
    plt.text(loadings[i, 0] * 1.1, loadings[i, 1] * 1.1, feature, color='b', ha='center', va='center')

plt.title('Correlation Circle (PC1 vs PC2)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.grid()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# Raw Loadings (Feature Contributions)
raw_loadings = pd.DataFrame(loadings,
                            columns=[f"PC{i+1}" for i in range(Vt.shape[0])],
                            index=numerical_cols)

# Plot grouped bar chart for raw loadings
plt.figure(figsize=(12, 8))
x = np.arange(len(raw_loadings.index))  # Position of attributes
width = 0.15  # Width of each bar

# Plot bars for each principal component
for i, pc in enumerate(raw_loadings.columns):
    plt.bar(x + i * width, raw_loadings[pc], width, label=pc)

# Add labels and title
plt.title('Raw Loadings of Each Attribute by Principal Component')
plt.xlabel('Attributes')
plt.ylabel('Raw Loadings')
plt.xticks(x + width * (len(raw_loadings.columns) - 1) / 2, raw_loadings.index)
plt.axhline(0, color='black', linestyle='--', linewidth=1)  # Add a horizontal line at y=0
plt.legend(title='Principal Components')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Explained Variance and Cumulative Variance
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.7, label='Explained Variance')
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o', color='red', label='Cumulative Variance')
plt.title('Explained Variance and Cumulative Variance (SVD)')
plt.xlabel('Number of Principal Components')
plt.ylabel('Variance Explained')
plt.legend(loc='best')
plt.grid()
plt.show()

# Summary statistics
summary_stats = X.describe().T  # Transpose for better readability
summary_stats['median'] = X.median()  # Add median column

# Print summary statistics
print("Summary Statistics:")
print(summary_stats)

