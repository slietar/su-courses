load output/cv/structure.pdb, CV
load output/sasa/structure.pdb, SASA
bg_color white
spectrum b, selection=CV
spectrum b, selection=SASA
disable CV
disable SASA
