from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.tree import DecisionTreeClassifier

from .residues import all_descriptors, native_descriptors, phenotypes
from .. import plots as _


# Clusering

clustering_df = native_descriptors[phenotypes.any(axis='columns')]

model = KMeans(n_clusters=4, random_state=0)
classes = model.fit_predict(scale(clustering_df))
classes_series = pd.Series(classes, index=clustering_df.index, name='class')

print(classes_series.value_counts().sort_index())

# print(model.inertia_)
# print(native_descriptors)
# print(classes)
# print(classes.shape)

# print(pd.DataFrame(model.cluster_centers_, columns=native_descriptors.columns))


# fig, ax = plt.subplots()

# ax.scatter(consolidated_residues.cv_10, consolidated_residues.rmsf, c=p, s=1.0)
# ax.grid()

# plt.show()


effects_df = phenotypes.join(classes_series, how='right')
effects_df.groupby('class').aggregate(np.mean).plot.bar(legend=False)

# print(df)

# fig, ax = plt.subplots()

# ax.bar(range(len(mutation_effects.columns)))



# Classification

classification_df = all_descriptors.loc[clustering_df.index]

classifier = DecisionTreeClassifier(max_depth=3, random_state=0)
classifier.fit(classification_df, classes)


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
                  feature=classification_df.columns[feature[i]],
                  threshold=threshold[i],
                  right=children_right[i],
                  value=values[i],
              )
          )


print('Score', classifier.score(classification_df, classes))
print_tree(classifier.tree_)


plt.show()


# Cluster 0
#  Gemme > -3.571396231651306 (not conserved)
#  CV10 > 0.5319907665252686 (center)
#  polymorphism < 2.6735538244247437 (not polymorphic)
#
# Cluster 2
#  Gemme > -3.571396231651306 (not conserved)
#  CV10 < 0.5319907665252686 (surface)
#  polymorphism < 3.3497501611709595 (not polymorphic)
#
# Cluster 3
#  Gemme < -3.571396231651306 (conserved)
#  polymorphism < 2.6735538244247437 (not polymorphic)
#  OR CV10 > 0.7173609137535095 (center)
