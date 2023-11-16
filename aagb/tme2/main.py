import numpy as np


M1 = np.array([[0,8,7,12], [8,0,9,14], [7,9,0,11], [12,14,11,0]])
M2 = np.array([[0,2,3,8,14,18],[2,0,3,8,14,18],
      [3,3,0,8,14,18],[8,8,8,0,14,18],
      [14,14,14,14,0,18],[18,18,18,18,18,0]])
#UPGMA
M3 = np.array([[0,19,27,8,33,18,13],[19,0,31,18,36,1,13],
          [27,31,0,26,41,32,29],[8,18,26,0,31,17,14],
          [33,36,41,31,0,35,28],[18,1,32,17,35,0,12],
          [13,13,29,14,28,12,0]])
#Neighbor Joining
M4 = np.array([[0,2,4,6,6,8],[2,0,4,6,6,8],
          [4,4,0,6,6,8],[6,6,6,0,4,8],
          [6,6,6,4,0,8],[8,8,8,8,8,0]])


def is_additive(matrix: np.ndarray):
  p = 0

  for a in range(len(matrix)):
    for b in range(a):
      for c in range(b):
        for d in range(c):
          x = matrix[a, b] + matrix[c, d]
          y = matrix[a, d] + matrix[b, c]
          z = matrix[a, c] + matrix[b, d]

          if not (x >= z and y >= z) and not (x >= y and z >= y) and not (y >= x and z >= x):
            return False

          p += 1

  return True


# print(is_additive(M1))
# print(is_additive(M2))
# print(is_additive(M3))
# print(is_additive(M4))


def is_ultrametric(matrix: np.ndarray):
  for a in range(len(matrix)):
    for b in range(a):
      for c in range(b):
        if matrix[a, c] > max(matrix[a, b], matrix[b, c]):
          return False

  return True

# print(is_ultrametric(M1))
# print(is_ultrametric(M2))
# print(is_ultrametric(M3))
# print(is_ultrametric(M4))


def upgma(matrix: np.ndarray):
  n = len(matrix)
  clusters = [[i] for i in range(n)]
  dist =[[matrix[i, j] for j in range(n)] for i in range(n)]
  newick=""

  while n > 1:
    # Find the minimum distance
    min_dist = np.inf
    for i in range(n):
      for j in range(i):
        if dist[i][j] < min_dist:
          min_dist = dist[i][j]
          min_i = i
          min_j = j

    # Create the new cluster
    new_cluster = clusters[min_i] + clusters[min_j]
    last_cluster_i= clusters[min_i]
    last_cluster_j= clusters[min_j]
    newick= str(newick)+"("+str(last_cluster_i)+":"+str(min_dist)+","+str(last_cluster_j)+":"+str(min_dist)+")"
    # Update the clusters
    clusters.pop(max(min_i, min_j))
    clusters.pop(min(min_i, min_j))
    clusters.append(new_cluster)
  

    # Update the distance matrix
    new_dist = dist
    for i in range(n):
      if i!=min_i and i!=min_j:
       new_dist[i].append((dist[min_i][i] * len(last_cluster_i) + dist[min_j][i] * len(last_cluster_j)) / (len(last_cluster_i) + len(last_cluster_j)))
    new_dist.append([])
    for i in range(n):
      if i!=min_i and i!=min_j:
       new_dist[n].append((dist[min_i][i] * len(last_cluster_i) + dist[min_j][i] * len(last_cluster_j)) / (len(last_cluster_i) + len(last_cluster_j)))
    new_dist[n].append(0)
    new_dist.pop(max(min_i, min_j))
    new_dist.pop(min(min_i, min_j))
    for i in range(n-2):
      del new_dist[i][max(min_i, min_j)]
      del new_dist[i][min(min_i, min_j)]
    dist = new_dist
    n -= 1


  return newick

mtest=np.array([[0,17,21,31,23],[17,0,30,34,21],[21,30,0,28,39],[31,34,28,0,43],[23,21,39,43,0]])
print(upgma(mtest))




