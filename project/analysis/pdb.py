# -*- coding:utf-8 -*-

from dataclasses import dataclass
import re
import sys
from typing import IO


@dataclass
class Atom:
  x: float
  y: float
  z: float

  residue_number: int
  residue_type: str
  temp_factor: float
  sse: str = 'X'


#EXPDTA    X-RAY D
#str->str
def parse_expdta(ligne):
	re_espaceFin=re.compile("[ ]*$")
	return re_espaceFin.sub('',ligne[10:79])

#REMARK 2 RESOLUTION. 1.74 ANGSTROMS.
def parse_resol(ligne):
	resol=float(ligne[23:30])
	return resol

#NUMMDL 20
#str->int
def parse_nummdl(ligne):
	return int(ligne[10:14])

#DBREF 2JHQ A 1 226 UNP Q9KPK8 UNG_VIBCH 1    226
#str->str
def getChainDBREF(ligne):
	chain = ligne[12]
	if chain == " ":
		chain ="-"

	return chain
#ATOM   1272  CA  MET A 145      16.640  22.247  -9.383  1.00 56.65           C
#str->str*Atom ou None
def getCAAtom(ligne):
	re_whitespaces = re.compile("[ ]+")
	resname = re_whitespaces.sub('',ligne[17:20])
	atomname = re_whitespaces.sub('',ligne[12:16])
	chain = ligne[21]
	if chain == " ":
		chain ="-"
	resnumber = int(ligne[22:26])
	x = float(ligne[30:38])
	y = float(ligne[38:46])
	z = float(ligne[46:54])
	#Autres champs
	#inscode = re_whitespaces.sub('',ligne[26])
	#atomnumber = int(ligne[6:11])
	#altloc = re_whitespaces.sub('',ligne[16])
	#occupancy = float(ligne[54:60])
	tempfactor = float(ligne[60:66])
	#element = re_whitespaces.sub('',ligne[76:78])
	#charge = re_whitespaces.sub('',ligne[78:80])
	if atomname=="CA":
		return chain, Atom(x,y,z,resnumber, resname, tempfactor)
	else:
		return chain, None

#HELIX 1 HAGLYA 86 GLYA 94 1 9
#str->[int]
def getISeqNumHelix(ligne):
	lesPosH=[]
	re_whitespaces = re.compile("[ ]+")
	#initResName = re_whitespaces.sub('',ligne[15:18])
	#initICode = re_whitespaces.sub('',ligne[25])
	#endResName = re_whitespaces.sub('',ligne[27:30])
	#endICode = re_whitespaces.sub('',ligne[37])
	#Class = int(ligne[38:40])
	#Length = int(ligne[71:76])
	initChainID = re_whitespaces.sub('',ligne[19])
	initSeqNum = int(ligne[21:25])
	endChainID = re_whitespaces.sub('',ligne[31])
	endSeqNum = int(ligne[33:37])
	if initChainID != endChainID:
 		print("Problème deux chaines différentes ? ", initChainID, endChainID)
 		print(ligne)
	else:
		for i in range(initSeqNum, endSeqNum+1):
			lesPosH.append(i)
	return lesPosH

#str->[int]
def getiSeqNumSheet(ligne):
	lesPosS=[]
	re_whitespaces = re.compile("[ ]+")
	#sheetID = re_whitespaces.sub('',ligne[11:14])
	#numStrands = int(ligne[14:16])
	#initResName = re_whitespaces.sub('',ligne[17:20])
	initChainID = re_whitespaces.sub('',ligne[21])
	initSeqNum = int(ligne[22:26])
	#initICode = re_whitespaces.sub('',ligne[26])
	endResName = re_whitespaces.sub('',ligne[28:31])
	endChainID = re_whitespaces.sub('',ligne[32])
	endSeqNum = int(ligne[33:37])
	#endICode = re_whitespaces.sub('',ligne[37])
	#sense = int(ligne[38:40])
	if initChainID != endChainID:
 		print("Problème deux chaines différentes ? ", initChainID, endChainID)
 		print(ligne)
	else:
		for i in range(initSeqNum, endSeqNum+1):
			lesPosS.append(i)
	return lesPosS

#[Atom]*[int]*[int]->[Atom]
#Attention lesAtomes est modifiée
def ajouterSSEAtomes(lesAtomes, lesH, lesS):
	if(len(lesH)==0 and len(lesS)==0):
		return lesAtomes
	lesH.sort()
	lesS.sort()
	iAt=0
	iH=0
	iS=0
	#3)lesAtomes
	print
	while iAt<len(lesAtomes) and (iH<len(lesH) or iS<len(lesS)):
		if iH<len(lesH) and lesAtomes[iAt].numRes==lesH[iH]:
			lesAtomes[iAt].sse="H"
			iH+=1
		elif iS<len(lesS) and lesAtomes[iAt].numRes==lesS[iS]:
			lesAtomes[iAt].sse="S"
			iS+=1
		iAt+=1
	return lesAtomes

def getNumModel(ligne):
	return int(ligne[10:14])

#str->str*float*int*[str]*[Atom]
#exp, resol, nummdl, lesChaines, lesAtomes
def read_pdb(f: IO[str], /, chain: str ="-", model: str = "1") -> tuple[str, float, int, None, list[Atom]]:
	#EXPDTA peut être sur plusieurs lignes
	#Pour avoir la première, j'ajoute des espaces pour les champs continuation (colonnes 9 - 10)
	re_expdta = re.compile("^EXPDTA {4}")
	#Format: REMARK puis 3 espaces puis 2 puis un espace puis RESOLUTION et ANGSTROM à la fin
	re_resolution = re.compile("^REMARK {3}2 RESOLUTION.*ANGSTROMS")
	re_atom = re.compile("^ATOM")
	re_dbref = re.compile("^DBREF")
	re_nummdl = re.compile("^NUMMDL")
	re_helix = re.compile("^HELIX")
	re_sheet = re.compile("^SHEET")
	re_model = re.compile("^MODEL")
	exp=None
	resol=None
	nbmdl=None
	numModel=None
	lesResSheet=[]
	lesResHelix=[]
	lesChaines=[]
	lesAtomes=[]
	for ligne in f:
		if re_expdta.search(ligne):
			exp=parse_expdta(ligne)
		elif re_resolution.search(ligne):
			resol=parse_resol(ligne)
		elif re_nummdl.search(ligne):
			nbmdl=parse_nummdl(ligne)
		elif re_dbref.search(ligne):
			lesChaines.append(getChainDBREF(ligne))
		elif re_model.search(ligne):
			numModel=getNumModel(ligne)
		elif re_atom.search(ligne):
			c,a=getCAAtom(ligne)
			#print modelVoulu, numModel, c, chaineVoulue, a
			if (numModel==None):
			    numModel=0
			if (a!=None and (numModel==0 or str(numModel)==model) and c==chain) :
				lesAtomes.append(a)
		elif re_helix.search(ligne):
			lesResHelix+=getISeqNumHelix(ligne)
		elif re_sheet.search(ligne):
			lesResHelix+=getiSeqNumSheet(ligne)

	lesAtomes=ajouterSSEAtomes(lesAtomes, lesResHelix, lesResSheet)

	return  exp, resol, nbmdl, lesChaines, lesAtomes

def write_pdb(fOUT: IO[str], listeAtomes: list[Atom], /):
	N = len(listeAtomes)
	for i in range(N):
		fOUT.write("ATOM  %5d  CA  %s  %4d    %8.3f%8.3f%8.3f%6.2f%6.2f            \n"%(i+1,listeAtomes[i].residue_type,listeAtomes[i].residue_number,listeAtomes[i].x,listeAtomes[i].y,listeAtomes[i].z,1.00,listeAtomes[i].temp_factor))
	fOUT.write("TER\n")
	fOUT.write("END\n")
	fOUT.close()

def main() :
	if (len(sys.argv)<2):
		print("USAGE: ", sys.argv[0], "<pdb file>")
		sys.exit(1)
	filename = sys.argv[1]
	exp, resol, nmdl, lesChaines, lesAtomes=readPDB(filename, "A")
	print(exp,resol, nmdl, lesChaines)
	print("Nb Atomes lus:", len(lesAtomes))
	#print lesAtomes[0].x

if __name__ == '__main__':
	main()
