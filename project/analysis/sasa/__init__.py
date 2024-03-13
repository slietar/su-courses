import freesasa
import pandas as pd

from .. import shared


structure = freesasa.Structure(str(shared.root_path / 'drive/FBN1_AlphaFold.pdb'))
result = freesasa.calc(structure)

residue_areas = [{
  'position': int(area.residueNumber),

  'apolar': area.relativeApolar,
  'main_chain': area.relativeMainChain,
  'polar': area.relativePolar,
  'side_chain': area.relativeSideChain,
  'total': area.relativeTotal,
} for area in result.residueAreas()['A'].values()]

sasa = pd.DataFrame.from_records(residue_areas, index='position')


__all__ = [
  'sasa'
]
