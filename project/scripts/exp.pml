load ../drive/FBN1_AlphaFold.pdb, FBN1
color green, %FBN1

fetch 1UZP
color orange, %1UZP
align 1UZP, FBN1

remove solvent

disable FBN1
disable 1UZP
