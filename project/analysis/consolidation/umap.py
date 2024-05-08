import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import scale
from sklearn.tree import DecisionTreeClassifier

from .. import plots as _
from .. import shared, utils
from .residues import all_descriptors, native_descriptors, phenotypes


# print(all_descriptors)
print(native_descriptors)

@utils.cache('umap')
def compute_umap():
  from umap import UMAP

  reducer = UMAP(random_state=0)
  embedding = reducer.fit_transform(scale(native_descriptors))

  return embedding


phenotypes_df = phenotypes.loc[all_descriptors.index]
pathogenic = phenotypes_df.any(axis='columns')

embedding = compute_umap()

clustering_model = KMeans(n_clusters=2, random_state=0)
# clustering_model = DBSCAN(eps=0.5, min_samples=5)
classes = clustering_model.fit_predict(embedding)
classes_series = pd.Series(classes, index=all_descriptors.index, name='class')

print(classes_series.value_counts().sort_index())


classifier = DecisionTreeClassifier(max_depth=1, random_state=0)
classifier.fit(all_descriptors, classes)


fig, axs = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True)
axs[0, 0].scatter(embedding[:, 0], embedding[:, 1], s=0.5)
axs[0, 0].set_title('Default')

axs[0, 1].scatter(embedding[:, 0], embedding[:, 1], c=pathogenic, s=0.5)
axs[0, 1].set_title('Pathogenic')

axs[1, 0].scatter(embedding[:, 0], embedding[:, 1], c=classes, s=0.5)
axs[1, 0].set_title('K means')

axs[1, 1].scatter(embedding[:, 0], embedding[:, 1], c=classifier.predict(all_descriptors), s=0.5)
axs[1, 1].set_title('Decision tree classifier')


for ax in axs.flat:
    ax.grid()


def print_tree(tree):
  n_nodes = tree.node_count
  children_left = tree.children_left
  children_right = tree.children_right
  feature = tree.feature
  threshold = tree.threshold
  values = tree.value

  node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
  is_leaves = np.zeros(shape=n_nodes, dtype=bool)
  stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
  while len(stack) > 0:
      # `pop` ensures each node is only visited once
      node_id, depth = stack.pop()
      node_depth[node_id] = depth

      # If the left and right child of a node is not the same we have a split
      # node
      is_split_node = children_left[node_id] != children_right[node_id]
      # If a split node, append left and right children and depth to `stack`
      # so we can loop through them
      if is_split_node:
          stack.append((children_left[node_id], depth + 1))
          stack.append((children_right[node_id], depth + 1))
      else:
          is_leaves[node_id] = True

  print(
      "The binary tree structure has {n} nodes and has "
      "the following tree structure:\n".format(n=n_nodes)
  )
  for i in range(n_nodes):
      if is_leaves[i]:
          print(
              "{space}node={node} is a leaf node with value={value}.".format(
                  space=node_depth[i] * "\t", node=i, value=np.argmax(values[i][0])
              )
          )
      else:
          print(
              "{space}node={node} is a split node: "
              "go to node {left} if {feature} <= {threshold} "
              "else to node {right}.".format(
                  space=node_depth[i] * "\t",
                  node=i,
                  left=children_left[i],
                  feature=all_descriptors.columns[feature[i]],
                  threshold=threshold[i],
                  right=children_right[i],
                  value=values[i],
              )
          )

print_tree(classifier.tree_)


with (shared.output_path / 'umap.png').open('wb') as file:
  fig.savefig(file)
