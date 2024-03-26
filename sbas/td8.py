from matplotlib import pyplot as plt
import numpy as np


data = np.array([1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# tpr = [(data).sum() for i in range(0, 19)]
# print(len(data))

tp = np.array([0, *np.cumsum(data)])
fp = np.array([0, *np.cumsum(1 - data)])

tpr = tp / tp[-1]
fpr = fp / fp[-1]

precision = tp / (tp + fp)
recall = tpr


fig, ax = plt.subplots()

ax.plot(fpr, tpr, marker='o')
ax.plot([0, 1], [0, 1], color='k', linestyle='--')

ax.set_xlabel('FPR')
ax.set_ylabel('TPR')

fig, ax = plt.subplots()

ax.plot(recall, precision, marker='o')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')

plt.show()
