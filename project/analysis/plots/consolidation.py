import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from .. import shared
from ..cv import cv
from ..dssp import dssp
from ..gemme import gemme_mean
from ..pae import pae_mean_by_position
from ..plddt import plddt
from ..polymorphism import polymorphism_score
from ..rmsf import rmsf_by_position
from ..sasa import sasa
from .utils import ProteinMap, set_colobar_label


fig, ax = plt.subplots(figsize=(25, 8))

map = ProteinMap(ax)


# DSSP

im_dssp = map.plot_dataframe(
  dssp.ss_contextualized.rename('Secondary structure')
)


# SASA

im_sasa = map.plot_dataframe(
  -sasa.total.rename('Total SASA')
)


# Circular variance

im_cv = map.plot_dataframe(
  pd.concat([
    cv.loc[:, 10.0].rename('CV (cutoff 10 Å)'),
    cv.loc[:, 20.0].rename('CV (cutoff 20 Å)')
  ], axis='columns')
)

# cbar = fig.colorbar(im_cv, ax=ax, pad=0.0)
# set_colobar_label(cbar, 'Circular variance')


# Mean GEMME

im_gemme = map.plot_dataframe(
  gemme_mean.rename('Mean GEMME')
)

# cbar = fig.colorbar(im_cv, ax=ax, pad=0.0)
# set_colobar_label(cbar, 'GEMME')


# pLDDT

im_plddt = map.plot_dataframe(
  plddt.alphafold_pruned.rename('pLDDT')
)

# cbar = fig.colorbar(im_plddt, ax=ax, pad=0.0)
# set_colobar_label(cbar, 'pLDDT')


# RMSF

im_rmsf = map.plot_dataframe(
  rmsf_by_position.rename('RMSF')
)

# cbar = fig.colorbar(im_plddt, ax=ax, pad=0.0)
# set_colobar_label(cbar, 'RMSF (Å)')


# PAE

im_pae = map.plot_dataframe(
  pae_mean_by_position.rename(columns=dict(
    pae_inter='Mean PAE over residues of neighboring domains',
    pae_intra='Mean PAE over residues of the domain itself'
  ))
)

# cbar = fig.colorbar(im_pae, ax=ax, pad=0.0)
# set_colobar_label(cbar, 'PAE')


# Polymorphism score

im_poly = map.plot_dataframe(
  polymorphism_score.rename('Polymorphism score')
)

# cbar = fig.colorbar(im_poly, ax=ax, pad=0.0)
# set_colobar_label(cbar, 'Polymorphism score')


# Output

map.finish()


with (shared.output_path / 'consolidation.png').open('wb') as file:
  fig.savefig(file)
