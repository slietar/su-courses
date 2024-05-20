from typing import IO

import pandas as pd


def load(file: IO[str], /):
  def process_line(line: str):
    return [item for item in [
			# "ATOM  "
      int(line[6:11]), # 5
			# " "
      line[12:16].strip(), # 4
      line[16].strip(), # 1
			# " "
      line[17:20].strip(), # 3
			# " "
      line[21], # 1
			# " "
      int(line[22:26]), # 4
      line[26].strip(), # 1
			# "    "
      float(line[30:38]), # 8
      float(line[38:46]), # 8
      float(line[46:54]), # 8
      float(line[54:60]), # 6
      float(line[60:66]), # 6
			# "      "
      line[72:76].strip(), # 4
      line[76:78].strip(), # 2
      line[78:80].strip() # 2
    ]]

  lines = (process_line(line) for line in file.readlines() if line.startswith('ATOM'))

  return pd.DataFrame(lines, columns=['atom_serial_number',
		'atom_name',
		'alt_loc_ind',
		'residue_name',
		'chain_id',
		'residue_seq_number',
		'code',
		'x',
		'y',
		'z',
		'occupancy',
		'temp_factor',
		'segment_id',
		'element_symbol',
		'charge'
  ]).set_index('atom_serial_number', drop=False)


def dump(atoms: pd.DataFrame, file: IO[str], /):
	for atom in atoms.itertuples():
		file.write(
      f'ATOM  {atom.atom_serial_number:<5} {atom.atom_name:4}{atom.alt_loc_ind} {atom.residue_name:3} {atom.chain_id}'
      f'{atom.residue_seq_number:<4}{atom.code}    {atom.x:8.3f}{atom.y:8.3f}{atom.z:8.3f}{atom.occupancy:6.2f}'
			f'{atom.temp_factor:6.2f}      {atom.segment_id:4}{atom.element_symbol:2}{atom.charge:2}\n'
    )

	file.write('HEADER                                                           46459b165d0e7114\n')
	file.write(f'TER{atoms.iloc[-1].atom_serial_number}      {atom.residue_name} {atom.chain_id}{atom.residue_seq_number:4}\n')

	file.write('TER\n')
	file.write('END\n')
