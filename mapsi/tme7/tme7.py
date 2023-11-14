from matplotlib import pyplot as plt
import numpy as np


def learnHMM(allX: np.ndarray, allS: np.ndarray, N: int, K: int):
  A = np.zeros((N, N), dtype=int)
  B = np.zeros((N, K), dtype=int)

  for prev_s, s in zip(allS[:-1], allS[1:]):
    A[prev_s, s] += 1

  for x, s in zip(allX, allS):
    B[s, x] += 1

  return (
    # Normalize both matrices
    A / np.sum(A, axis=1, keepdims=True),
    B / np.sum(B, axis=1, keepdims=True)
  )


# N = number of states
# K = number of possible observations
# T = number of observations in the sequence
#
# allx: (T,)
# Pi: (N,)
# A: (N, N)
# B: (N, K) -> given hidden state s, probability of observing x
#
def viterbi(all_x: np.ndarray, Pi: np.ndarray, A: np.ndarray, B: np.ndarray):
  N, K = B.shape
  T = len(all_x)

  # Initialize delta and psi
  delta = np.zeros((N, T))
  delta[:, 0] = np.log(Pi) + np.log(B[:, all_x[0]])

  psi = np.zeros((N, T), dtype=int)
  psi[:, 0] = -1

  # Iterate over all observation
  for t, x in enumerate(all_x):
    # Ignore the first observation since it was already processed during initialization
    if t < 1:
      continue

    delta[:, t] = (delta[:, t - 1] + np.log(A.T)).max(axis=1) + np.log(B[:, x])
    psi[:, t] = (delta[:, t - 1] + np.log(A.T)).argmax(axis=1)

    # Display 1 in 100000 delta values
    if t % 100000 == 0:
      print(f"{t=} {delta[:, t]=}")

  all_s = np.zeros(T, dtype=int)

  S = delta[:, -1].max()
  all_s[-1] = delta[:, -1].argmax()

  # Find the most likely path from the last state to the first
  for t in range(T - 2, -1, -1):
    all_s[t] = psi[all_s[t + 1], t + 1]

  return all_s


def get_and_show_coding(all_s_pred: np.ndarray, all_s: np.ndarray):
  all_s2 = np.where(all_s == 0, 0, 1)
  all_s_pred2 = np.where(all_s_pred == 0, 0, 1)

  fig, ax = plt.subplots(figsize=(15,2))
  ax.plot(all_s_pred2[100000:200000], label="prediction", ls="--")
  ax.plot(all_s2[100000:200000], label="annotation", lw=3, color="black", alpha=.4)
  plt.legend(loc="best")
  plt.show()

  return all_s_pred2, all_s2

def create_confusion_matrix(all_s_pred: np.ndarray, all_s: np.ndarray):
  N = len(np.unique(all_s))
  confusion_matrix = np.zeros((N, N), dtype=int)

  for s_pred, s in zip(all_s_pred, all_s):
    # We're using 1 - s and 1 - s_pred because the states are inverted
    confusion_matrix[1 - s, 1 - s_pred] += 1

  return confusion_matrix

"""
Le modèle est très mauvais pour prédire les non-codants, le taux de faux négatifs est plus élevé que celui de vrai négatifs.
"""

def create_seq(N,Pi,A,B,states,obs):
  seq_s = np.zeros(N, dtype=int)
  seq_x = np.zeros(N, dtype=int)
  seq_s[0] = np.random.choice(states, p=Pi)
  seq_x[0] = np.random.choice(range(len(obs)), p=B[seq_s[0], :])

  for i in range(1, N):
    seq_s[i] = np.random.choice(states, p=A[seq_s[i - 1]])
    seq_x[i] = np.random.choice(range(len(obs)), p=B[seq_s[i], :])

  for s, x in zip(seq_s, seq_x):
    print(f"{s} {obs[x]}")


def get_annoatation2(all_s: np.ndarray):
  all_s2 = np.zeros_like(all_s)
  index = 1

  while index < len(all_s):
    prev_s = all_s[index - 1]
    s = all_s[index]

    # If there is a gene start
    if prev_s == 0 and s == 1:
      all_s2[index] = 1
      all_s2[index + 1] = 2
      all_s2[index + 2] = 3
      index += 3

    # If there is a gene end
    elif prev_s == 3 and s == 0:
      all_s2[index - 3] += 3 # +3 to make it 4, 5 or 6
      all_s2[index - 2] += 3
      all_s2[index - 1] += 3
      index += 1

    # If we're inside a gene
    elif s != 0:
      all_s2[index] = s + 3
      index += 1

    # If we're outside a gene
    else:
      index += 1

  return all_s2
