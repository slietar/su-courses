# Antoine GRISLAIN, Simon LIETAR

from random import random
import math

from matplotlib import pyplot as plt
from pyAgrum import Potential
import numpy as np


def bernoulli(p: float):
  return random() < p

def binomiale(n: int, p: float):
  return sum(bernoulli(p) for _ in range(n))

def galton(l: int, n: int, p: float):
  return [binomiale(n, p) for _ in range(l)]

def histo_galton(l: int, n: int, p: float):
  plt.hist(galton(l, n, p), bins=range(n + 1))

def normale(k: int, sigma: float):
  if k % 2 == 0:
    raise ValueError("k doit être impair")

  return 1 / sigma / math.sqrt(2 * math.pi) * np.exp(-0.5 * (np.linspace(-2 * sigma, 2 * sigma, k) / sigma) ** 2)

def proba_affine(k: int, slope: float):
  if k % 2 == 0:
    raise ValueError("k doit être impair")
  if abs(slope) > (max_slope := 2 / k ** 2):
    raise ValueError(f"slope doit être inférieur à {max_slope}")

  return 1 / k + (np.arange(k) - (k - 1) / 2) * slope

def Pxy(PA: np.ndarray, PB: np.ndarray):
  return np.outer(PA, PB)

def calcYZ(P_XYZT: np.ndarray):
  return P_XYZT.sum(axis=(0, 3))

def calcXTcondYZ(P_XYZT: np.ndarray):
  return P_XYZT / calcYZ(P_XYZT)[None, :, :, None]

def calcX_etTcondYZ(P_XYZT: np.ndarray):
  YZ = calcYZ(P_XYZT)

  return (
    P_XYZT.sum(axis=3) / YZ[None, :, :],
    np.moveaxis(P_XYZT.sum(axis=0) / YZ[:, :, None], 2, 0)
  )

def testXTindepCondYZ(P_XYZT: np.ndarray, *, epsilon: float):
  XcondYZ, TcondYZ = calcX_etTcondYZ(P_XYZT)
  return np.allclose(calcXTcondYZ(P_XYZT), XcondYZ[:, :, :, None] * np.moveaxis(TcondYZ, 0, 2)[None, :, :, :], atol=epsilon, rtol=0.0)

def testXindepYZ(P_XYZT: np.ndarray, *, epsilon: float):
  P_XYZ = P_XYZT.sum(axis=3)
  P_X = P_XYZ.sum(axis=(1, 2))
  P_YZ = P_XYZ.sum(axis=0)

  return np.allclose(P_XYZ, P_X[:, None, None] * P_YZ[None, :, :], atol=epsilon, rtol=0.0)

def conditional_indep(potential: Potential, a: str, b: str, cond: list[str], *, epsilon: float):
  P_cond = potential / (potential.margSumIn(cond) if cond else 1)

  return (P_cond.margSumIn([a, b, *cond]) - P_cond.margSumIn([a, *cond]) * P_cond.margSumIn([b, *cond])).abs().max() < epsilon

def compact_conditional_proba(potential: Potential, target_var: str):
  vars = set(potential.names) - {target_var}

  for var in vars.copy():
    if conditional_indep(potential, var, target_var, list(vars - {var}), epsilon=1e-5):
      vars.remove(var)

  return (potential / potential.margSumIn(list(vars))).margSumIn([target_var, *vars]).putFirst(target_var)

def create_bayesian_network(input_potential: Potential):
  potential = input_potential
  potentials = list[Potential]()

  for var in input_potential.names:
    q = compact_conditional_proba(potential, var)
    potential = potential / q
    potentials.append(q)

  return potentials

def calcNbParams(potential: Potential):
  return potential.domainSize(), sum([p.domainSize() for p in create_bayesian_network(potential)])
