import numpy as np


def distance(alpha: list[str], match_score: float, mismatch_score: float):
  return np.ones((len(alpha), len(alpha))) * mismatch_score + (match_score - mismatch_score) * np.identity(len(alpha))

def align_nw(seq1: list[str], seq2: list[str]):
  alpha = list(set(seq1) | set(seq2))
  seq1_n = [alpha.index(x) for x in seq1]
  seq2_n = [alpha.index(x) for x in seq2]

  gap_score_open = -2
  gap_score_extension = -1

  distance_matrix = distance(alpha, match_score=1, mismatch_score=-2)
  score = np.zeros((len(seq1) + 1, len(seq2) + 1))
  fleche= np.zeros((len(seq1) + 1, len(seq2) + 1), dtype=np.uint8)

  score[0, :] = np.arange(0, -score.shape[1], step=-1)
  score[:, 0] = np.arange(0, -score.shape[0], step=-1)

  fleche[0, 1:] = 1
  fleche[1:, 0] = 2

  for index1, letter1 in enumerate(seq1_n, start=1):
    for index2, letter2 in enumerate(seq2_n, start=1):
      match_score = distance_matrix[letter1, letter2]

      a = score[index1, index2 - 1] + (gap_score_extension if fleche[index1, index2 - 1] == 1 else gap_score_open)
      b = score[index1 - 1, index2] + (gap_score_extension if fleche[index1 - 1, index2] == 2 else gap_score_open)
      c = score[index1 - 1, index2 - 1] + match_score
      m = max(a, b, c)

      score[index1, index2] = m
      fleche[index1, index2] = ((1 if a == m else 0) << 0) + ((1 if b == m else 0) << 1) + ((1 if c == m else 0) << 2)


  position = (len(seq1), len(seq2))
  seq1_a = list[str]()
  seq2_a = list[str]()

  while position != (0, 0):
    if (fleche[position] & 1) != 0:
      seq1_a.append('-')
      seq2_a.append(seq2[position[1] - 1])
      position = (position[0], position[1] - 1)
    elif (fleche[position] & 2) != 0:
      seq1_a.append(seq1[position[0] - 1])
      seq2_a.append('-')
      position = (position[0] - 1, position[1])
    elif (fleche[position] & 4) != 0:
      seq1_a.append(seq1[position[0] - 1])
      seq2_a.append(seq2[position[1] - 1])
      position = (position[0] - 1, position[1] - 1)
    else:
      raise Exception

  return str().join(list(reversed(seq1_a))), str().join(list(reversed(seq2_a)))

print(align_nw(
  list('TCTGAAC'),
  list('CATGAC')
))
# print(distance(['A', 'C', 'G', 'T'], 1, -1))
