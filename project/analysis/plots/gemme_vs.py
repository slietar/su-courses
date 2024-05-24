from matplotlib import pyplot as plt

from ..gemme import gemme_all, gemme_orthologs, gemme_mutations


fig, ax = plt.subplots()

# ax.scatter(gemme_all.array.flat, (gemme_all.array - gemme_orthologs.array).flat, alpha=0.1, marker='.', s=1.0)
# ax.scatter(gemme_all.array.flat, (gemme_all.array - gemme_orthologs.array).flat, alpha=0.1, marker='.', s=1.0)
# ax.scatter(gemme_mutations.gemme_all, gemme_mutations.gemme_orthologs, marker='.', s=1.0)
ax.scatter(gemme_mutations.gemme_all, gemme_mutations.gemme_all - gemme_mutations.gemme_orthologs, marker='.', s=1.0)

plt.show()
