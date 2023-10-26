from pathlib import Path
from typing import Optional, Sequence, TypeVar
import urllib.request
import numpy as np


def base_distance_matrix(alphabet: list[str], *, match_score: float, mismatch_score: float):
  return np.ones((len(alphabet), len(alphabet)), dtype=int) * mismatch_score + (match_score - mismatch_score) * np.identity(len(alphabet), dtype=int)

def _fetch_rcsb_sequence(name: str):
  response = urllib.request.urlopen(f'https://www.rcsb.org/fasta/entry/{name}')

  while raw_line := response.readline().decode():
    line = raw_line.rstrip()

    if not line.startswith('>'):
      return line

  raise RuntimeError('No sequence found')

def _pack(values: Sequence[bool]):
  return sum((1 << i) if values[i] else 0 for i in range(len(values)))


T = TypeVar('T', str, list)

def _partition(items: T, size: int) -> list[T]:
  return [items[index:(index + size)] for index in range(0, len(items), size)]


def align_nw(
  seq1: Sequence[str],
  seq2: Sequence[str],
  *,
  alphabet: Optional[list[str]] = None,
  distance_matrix: Optional[np.ndarray] = None,
  gap_score_open: int = -2,
  gap_score_extension: int = -1
):
  alphabet = list(set(seq1) | set(seq2)) if alphabet is None else alphabet
  distance_matrix = base_distance_matrix(alphabet, match_score=1, mismatch_score=-2) if distance_matrix is None else distance_matrix

  seq1_n = [alphabet.index(x) for x in seq1]
  seq2_n = [alphabet.index(x) for x in seq2]

  # Score matrix construction

  score = np.zeros((len(seq1) + 1, len(seq2) + 1))
  arrows= np.zeros((len(seq1) + 1, len(seq2) + 1), dtype=np.uint8)

  score[0, :] = np.arange(0, -score.shape[1], step=-1)
  score[:, 0] = np.arange(0, -score.shape[0], step=-1)

  arrows[0, 1:] = 1
  arrows[1:, 0] = 2

  for index1, letter1 in enumerate(seq1_n, start=1):
    for index2, letter2 in enumerate(seq2_n, start=1):
      match_score = distance_matrix[letter1, letter2]

      a = score[index1, index2 - 1] + (gap_score_extension if (arrows[index1, index2 - 1] & 1) != 0 else gap_score_open)
      b = score[index1 - 1, index2] + (gap_score_extension if (arrows[index1 - 1, index2] & 2) != 0 else gap_score_open)
      c = score[index1 - 1, index2 - 1] + match_score
      m = max(a, b, c)

      score[index1, index2] = m
      arrows[index1, index2] = _pack([a == m, b == m, c == m])

  # Traceback

  position = (len(seq1), len(seq2))
  seq1_a = list[str]()
  seq2_a = list[str]()

  while position != (0, 0):
    if (arrows[position] & 1) != 0:
      seq1_a.append('-')
      seq2_a.append(seq2[position[1] - 1])
      position = (position[0], position[1] - 1)
    elif (arrows[position] & 2) != 0:
      seq1_a.append(seq1[position[0] - 1])
      seq2_a.append('-')
      position = (position[0] - 1, position[1])
    elif (arrows[position] & 4) != 0:
      seq1_a.append(seq1[position[0] - 1])
      seq2_a.append(seq2[position[1] - 1])
      position = (position[0] - 1, position[1] - 1)
    else:
      raise RuntimeError

  # print('\n'.join(' '.join(('←' if (y & 1) else ('↑' if (y & 2) else '↖︎')) for y in x) for x in arrows))
  # print('\n'.join('|'.join((('←' if (y & 1) else ' ') + ('↑' if (y & 2) else ' ') + ('↖︎' if (y & 4) else ' ')) for y in x) for x in arrows))

  return str().join(list(reversed(seq1_a))), str().join(list(reversed(seq2_a)))

def format_alignment(seq1: str, seq2: str, label1: str, label2: str, *, width: int = 80):
  label_size = max(len(label1), len(label2))
  format_match = lambda a, b: ('|' if a == b else ':') if (a != '-' and b != '-') else ' '

  return '\n\n\n'.join([
    f'''{label1:{label_size}} {line1}\n{' ' * label_size} {str().join(format_match(x, y) for x, y in zip(line1, line2))}\n{label2:{label_size}} {line2}''' for line1, line2 in zip(_partition(seq1, width), _partition(seq2, width))
  ])

# print(align_nw(
#   list('TCTGAAC'),
#   list('CATGAC')
# ))

# print(distance(['A', 'C', 'G', 'T'], 1, -1))

# print(a)
# print(_partition(list(a), 80))

print('Needleman-Wunsch')

with Path('blosum62.txt').open('rt') as file:
  blosum62_alphabet = file.readline()[0:-1].split(' ')
  blosum62_matrix = np.loadtxt(file, delimiter=' ', dtype=int)

a, b = align_nw(
  list(_fetch_rcsb_sequence('2ABL')),
  list(_fetch_rcsb_sequence('1OPK')),
  alphabet=blosum62_alphabet,
  distance_matrix=blosum62_matrix,
  # distance_matrix=np.random.randint(-5, 1, blosum62_matrix.shape),
  gap_score_open=-11,
  gap_score_extension=-1
)

print(format_alignment(a, b, '2ABL', '1OPK'))


# SMITH-WATERMAN

def align_sw(seq1: Sequence[str], seq2: Sequence[str], *,
             alphabet: Optional[list[str]] = None,
             distance_matrix: Optional[np.ndarray] = None,
             gap_score_open: int = -2,
             gap_score_extension: int = -1):
    alphabet = list(set(seq1) | set(seq2)) if alphabet is None else alphabet
    distance_matrix = base_distance_matrix(alphabet, match_score=1, mismatch_score=-2) if distance_matrix is None else distance_matrix

    seq1_n = [alphabet.index(x) for x in seq1]
    seq2_n = [alphabet.index(x) for x in seq2]

    # Score matrix construction
    score = np.zeros((len(seq1) + 1, len(seq2) + 1))
    arrows = np.zeros((len(seq1) + 1, len(seq2) + 1), dtype=np.uint8)

    score[0, :] = np.arange(0, -score.shape[1], step=-1)
    score[:, 0] = np.arange(0, -score.shape[0], step=-1)

    arrows[0, 1:] = 1
    arrows[1:, 0] = 2

    max_score = 0  # Track the maximum score
    max_position = (0, 0)  # Track the position of the maximum score

    for index1, letter1 in enumerate(seq1_n, start=1):
        for index2, letter2 in enumerate(seq2_n, start=1):
            match_score = distance_matrix[letter1, letter2]

            a = score[index1, index2 - 1] + (gap_score_extension if (arrows[index1, index2 - 1] & 1) != 0 else gap_score_open)
            b = score[index1 - 1, index2] + (gap_score_extension if (arrows[index1 - 1, index2] & 2) != 0 else gap_score_open)
            c = score[index1 - 1, index2 - 1] + match_score
            m = max(a, b, c, 0)  # Ensure no negative scores

            score[index1, index2] = m
            arrows[index1, index2] = _pack([a == m, b == m, c == m])

            # Update the maximum score and its position
            if m > max_score:
                max_score = m
                max_position = (index1, index2)

    # Traceback for local alignment
    position = max_position
    seq1_a = list[str]()
    seq2_a = list[str]()

    while score[position] > 0:
        if (arrows[position] & 1) != 0:
            seq1_a.append('-')
            seq2_a.append(seq2[position[1] - 1])
            position = (position[0], position[1] - 1)
        elif (arrows[position] & 2) != 0:
            seq1_a.append(seq1[position[0] - 1])
            seq2_a.append('-')
            position = (position[0] - 1, position[1])
        elif (arrows[position] & 4) != 0:
            seq1_a.append(seq1[position[0] - 1])
            seq2_a.append(seq2[position[1] - 1])
            position = (position[0] - 1, position[1] - 1)
        else:
            raise RuntimeError

    # Reverse the sequences
    seq1_a = seq1_a[::-1]
    seq2_a = seq2_a[::-1]

    return (str().join(seq1_a), str().join(seq2_a)), (position[0] - 1, position[1] - 1)


# Align sequences and print the formatted result

print('\nSmith-Waterman')

(a, b), position = align_sw(
    list(_fetch_rcsb_sequence('2ABL')),
    list(_fetch_rcsb_sequence('1OPK')),
    alphabet=blosum62_alphabet,
    distance_matrix=blosum62_matrix,
    gap_score_open=-11,
    gap_score_extension=-1
)

print('Starting position on 2ABL:', position[0])
print('Starting position on 1OPK:', position[1])
print(format_alignment(a, b, '2ABL', '1OPK'))
