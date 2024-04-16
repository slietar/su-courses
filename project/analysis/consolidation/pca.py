from matplotlib import pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA


from .residues import consolidated_residues


df = consolidated_residues.dropna()

print(df)

model = PCA(n_components=2)
pc = model.fit_transform(df)

fig, ax = plt.subplots()

ax.scatter(pc[:, 0], pc[:, 1])
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')

plt.show()
