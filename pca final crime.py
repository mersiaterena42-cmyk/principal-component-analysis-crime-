

from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

crime = fetch_ucirepo(id=183)

X = crime.data.features.copy()
y = crime.data.targets['ViolentCrimesPerPop'].copy()

X = X.select_dtypes(include=[np.number])

print("\nORIGINAL DATA SHAPE")
print("Rows (communities):", X.shape[0])
print("Columns (variables):", X.shape[1])

print("\nVARIABLE NAMES")
for i, name in enumerate(X.columns, start=1):
    print(i, name)

print("\nFIRST 10 ROWS OF ORIGINAL DATA")
print(X.head(10))

print("\nMISSING VALUES")
missing_values = X.isna().sum()
print(missing_values)

missing_percent = X.isna().mean() * 100

X = X.loc[:, missing_percent < 20]

X = X.fillna(X.mean())

print("\nNUMBER OF VARIABLES AFTER CLEANING")
print(X.shape[1])

sumX = X.sum()

print("\nSTEP 1: SUM OF EACH VARIABLE")
print(sumX)

meanX = X.mean()

print("\nSTEP 2: MEAN OF EACH VARIABLE")
print(meanX)

X_centered = X - meanX

print("\nSTEP 3: CENTRED DATA")
print(X_centered.head(10))

stdX = X.std(ddof=1)

print("\nSTEP 4: STANDARD DEVIATION")
print(stdX)

non_constant = stdX > 0

X = X.loc[:, non_constant]
X_centered = X_centered.loc[:, non_constant]
stdX = stdX[non_constant]

Z = X_centered / stdX

print("\nSTEP 5: Z-SCORES")
print(Z.head(10))

C = Z.cov()

print("\nSTEP 6: COVARIANCE MATRIX")
print(C)

eigenvalues, eigenvectors = np.linalg.eigh(C)

order = np.argsort(eigenvalues)[::-1]

eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]

print("\nSTEP 7: EIGENVALUES")
print(eigenvalues)

eigenvector_table = pd.DataFrame(
    eigenvectors,
    index=X.columns,
    columns=[f'PC{i+1}' for i in range(len(eigenvalues))]
)

print("\nSTEP 8: EIGENVECTORS")
print(eigenvector_table)

explained = eigenvalues / eigenvalues.sum() * 100

print("\nSTEP 9: EXPLAINED VARIANCE (%)")
print(explained)

cumulative_explained = np.cumsum(explained)

print("\nSTEP 10: CUMULATIVE EXPLAINED VARIANCE (%)")
print(cumulative_explained)

summary = pd.DataFrame({
    'PC': [f'PC{i+1}' for i in range(len(eigenvalues))],
    'Eigenvalue': eigenvalues,
    'Explained %': explained,
    'Cumulative %': cumulative_explained
})

print("\nSTEP 11: PCA SUMMARY")
print(summary)

PC_scores = Z.to_numpy() @ eigenvectors

PC_table = pd.DataFrame(
    PC_scores,
    columns=[f'PC{i+1}' for i in range(PC_scores.shape[1])]
)

print("\nSTEP 12: PRINCIPAL COMPONENT SCORES")
print(PC_table.head(10))

result = pd.DataFrame({
    'PC1': PC_table['PC1'],
    'PC2': PC_table['PC2'],
    'PC3': PC_table['PC3'],
    'ViolentCrime': y.reset_index(drop=True)
})

print("\nSTEP 13: FIRST 3 PCS WITH VIOLENT CRIME")
print(result.head(10))

num_pc_90 = np.argmax(cumulative_explained >= 90) + 1
num_pc_95 = np.argmax(cumulative_explained >= 95) + 1

print("\nSTEP 14: DIMENSIONALITY REDUCTION")
print("Original variables:", X.shape[1])
print("PCs needed for 90% variance:", num_pc_90)
print("PCs needed for 95% variance:", num_pc_95)

pc1 = eigenvector_table['PC1'].abs().sort_values(ascending=False)

print("\nSTEP 15: TOP VARIABLES CONTRIBUTING TO PC1")
print(pc1.head(15))

pc2 = eigenvector_table['PC2'].abs().sort_values(ascending=False)

print("\nSTEP 16: TOP VARIABLES CONTRIBUTING TO PC2")
print(pc2.head(15))

pc3 = eigenvector_table['PC3'].abs().sort_values(ascending=False)

print("\nSTEP 17: TOP VARIABLES CONTRIBUTING TO PC3")
print(pc3.head(15))