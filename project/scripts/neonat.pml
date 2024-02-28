load ../drive/FBN1_AlphaFold.pdb, neonat_full
load ../domains/neonat/Néonat_ef370_0_unrelaxed_rank_002_alphafold2_ptm_model_1_seed_000.pdb, neonat
color orange, %neonat
color green, %neonat_full
color red, %neonat_full and resi 952-1362
alter %neonat, resi=str(int(resi)+951)
color marine, (%neonat or %neonat_full) and resi 1053+1048+817+1032+1068+1073+1322+1117+1086+1032+1254+1073+1302+1212+1097+1070+1111+1249+1068+1074+1073+1039

disable neonat
disable neonat_full
