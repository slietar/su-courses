% Données

equiv(sculpteur, and(personne, some(aCree, sculpture))).
equiv(auteur, and(personne, some(aEcrit, livre))).
equiv(editeur, and(personne, and(not(some(aEcrit, livre)), some(aEdite, livre)))).
equiv(parent, and(personne, some(aEnfant, anything))).

cnamea(personne).
cnamea(livre).
cnamea(objet).
cnamea(sculpture).
cnamea(anything).
cnamea(nothing).

cnamea(anything).
cnamea(nothing).

cnamena(auteur).
cnamena(editeur).
cnamena(sculpteur).
cnamena(parent).

iname(michelAnge).
iname(david).
iname(sonnets).
iname(vinci).
iname(joconde).

rname(aCree).
rname(aEcrit).
rname(aEdite).
rname(aEnfant).

inst(michelAnge, personne).
inst(david, sculpture).
inst(sonnets, livre).
inst(vinci, personne).
inst(joconde, objet).

instR(michelAnge, david, aCree).
instR(michelAnge, sonnets, aEcrit).
instR(vinci, joconde, aCree).

init_Tbox([
  (sculpteur, and(personne, some(aCree, sculpture))),
  (auteur, and(personne, some(aEcrit, livre))),
  (editeur, and(personne, and(not(some(aEcrit, livre)), some(aEdite, livre)))),
  (parent, and(personne, some(aEnfant, anything)))
]).

init_Abi([
  (michelAnge, personne),
  (david, sculpture),
  (sonnets, livre),
  (vinci, personne),
  (joconde, objet)
]).

init_Abr([
  (michelAnge, david, aCree),
  (michelAnge, sonnet, aEcrit),
  (vinci, joconde, aCree)
]).


% Vérifient le contenu d'une Abox ou Tbox.
% Exemples:
%   check_Tbox([(sculpteur,and(personne,some(aCree,sculpture))), (auteur,and(personne,some(aEcrit,livre))), (editeur,and(personne,and(not(some(aEcrit,livre)),some(aEdite,livre)))), (parent,and(personne,some(aEnfant,anything)))]).
%     -> true
%   check_Abi([(michelAnge,personne), (david,sculpture), (sonnets, livre), (vinci,personne), (joconde,objet)]).
%     -> true
%   check_Abr([(michelAnge, david, aCree), (michelAnge, sonnets, aEcrit),(vinci, joconde, aCree)]).
%     -> true

check_Tbox([]).
check_Tbox([(X, Y) | A]) :- cnamena(X), concept(Y), check_Tbox(A).

check_Abi([]).
check_Abi([(X, Y) | A]) :- iname(X), cnamea(Y), check_Abi(A).

check_Abr([]).
check_Abr([(X, Y, Z) | A]) :- iname(X), iname(Y), rname(Z), check_Abr(A).


% Teste si un objet est un concept.
% Exemples:
%   concept(personne)
%     -> true
%   concept(auteur)
%     -> true
%   concept(and(personne, livre))
%     -> true
%   concept(some(aCree, sculpture))
%     -> true
%   concept(michelAnge)
%     -> false

concept(X) :- cnamea(X).
concept(X) :- cnamena(X).
concept(and(X, Y)) :- concept(X), concept(Y).
concept(or(X, Y)) :- concept(X), concept(Y).
concept(not(X)) :- concept(X).
concept(some(R, C)) :- rname(R), concept(C).
concept(all(R, C)) :- rname(R), concept(C).


% Teste si le concept dans le second argument ne référence pas le premier. Le premier argument doit être un concept non atomique.
% Exemples:
%   not_autoref(auteur, and(personne, some(aEcrit, livre)))
%     -> true
%   not_autoref(auteur, or(auteur, editeur))
%     -> false

not_autoref(N, and(X, Y)) :- not_autoref(N, X), not_autoref(N, Y).
not_autoref(N, or(X, Y)) :- not_autoref(N, X), not_autoref(N, Y).
not_autoref(_, X) :- cnamea(X).
not_autoref(N, X) :- cnamena(X), N \== X.
not_autoref(N, not(X)) :- not_autoref(N, X).
not_autoref(N, some(_, C)) :- not_autoref(N, C).
not_autoref(N, all(_, C)) :- not_autoref(N, C).


% Teste si un concept non atomique se référence lui-même dans son expression.

autoref(N, X) :- \+ not_autoref(N, X).


% Met le premier argument sous forme normale négative dans le second.
% Exemples:
%   nnf(not(and(a, b)), X)
%     -> X = or(not(a), not(b))
%   nnf(not(some(a, b)), X)
%     -> X = all(a, not(b))

nnf(not(and(C1,C2)),or(NC1,NC2)):- nnf(not(C1),NC1),
nnf(not(C2),NC2),!.
nnf(not(or(C1,C2)),and(NC1,NC2)):- nnf(not(C1),NC1),
nnf(not(C2),NC2),!.
nnf(not(all(R,C)),some(R,NC)):- nnf(not(C),NC),!.
nnf(not(some(R,C)),all(R,NC)):- nnf(not(C),NC),!.
nnf(not(not(X)),Y):- nnf(X,Y),!.
nnf(not(X),not(X)):-!.
nnf(and(C1,C2),and(NC1,NC2)):- nnf(C1,NC1),nnf(C2,NC2),!.
nnf(or(C1,C2),or(NC1,NC2)):- nnf(C1,NC1), nnf(C2,NC2),!.
nnf(some(R,C),some(R,NC)):- nnf(C,NC),!.
nnf(all(R,C),all(R,NC)) :- nnf(C,NC),!.
nnf(X,X).


% Remplace les concepts complexes par leurs expression respectives.
% Exemples:
%   repl_equiv(auteur, X)
%     -> X = and(personne, some(aEcrit, livre))
%   repl_equiv(and(auteur, sculpteur), X)
%     -> X = and(and(personne, some(aEcrit, livre)), and(personne, some(aCree, sculpture)))

repl_equiv(and(X, Y), and(RX, RY)) :- repl_equiv(X, RX), repl_equiv(Y, RY), !.
repl_equiv(or(X, Y), or(RX, RY)) :- repl_equiv(X, RX), repl_equiv(Y, RY), !.
repl_equiv(not(X), not(Y)) :- repl_equiv(X, Y), !.
repl_equiv(some(R, X), some(R, RX)) :- repl_equiv(X, RX), !.
repl_equiv(all(R, X), all(R, RX)) :- repl_equiv(X, RX), !.
repl_equiv(A, B) :- equiv(A, B), !.
repl_equiv(A, A).


% Remplace les expressions complexes d'une Tbox et met toutes les expressions sous forme normale négative.
% Exemples:
%   traitement_Tbox([(editeur, and(personne, and(not(some(aEcrit, livre)), some(aEdite, livre))))], X)
%     -> X = [(editeur, and(personne, and(all(aEcrit, not(livre)), some(aEdite, livre))))]

traitement_Tbox([], []).
traitement_Tbox([(N, A) | T], [(N, RA) | RT]) :- traitement_Tbox(T, RT), repl_equiv(A, B), nnf(B, RA).


% Remplace une valeur dans une expression par sa définition.
% Paramètres:
%   - expression sans remplacement
%   - nom à remplacer
%   - expression à utiliser pour remplacer
%   - expression avec remplacement
% Exemples:
%   repl_def(and(a, b), b, c, X)
%     -> X = and(a, c)

repl_def(and(X, Y), N, D, and(RX, RY)) :- repl_def(X, N, D, RX), repl_def(Y, N, D, RY), !.
repl_def(or(X, Y), N, D, or(RX, RY)) :- repl_def(X, N, D, RX), repl_def(Y, N, D, RY), !.
repl_def(not(X), N, D, not(RX)) :- repl_def(X, N, D, RX), !.
repl_def(some(R, X), N, D, some(R, RX)) :- repl_def(X, N, D, RX), !.
repl_def(all(R, X), N, D, all(R, RX)) :- repl_def(X, N, D, RX), !.
repl_def(N, N, D, D) :- !.
repl_def(X, _, _, X).


% Remplace plusieurs définitions.
% Paramètres:
%   - expression sans remplacement
%   - liste des noms et expressions pour le remplacement
%   - expression avec remplacement
% Exemples:
%   repl_defs(and(a, b), [(a, ap), (b, bp)], X)
%     -> X = and(ap, bp)

repl_defs(X, [], X).
repl_defs(X, [(N, D) | T], RX) :- repl_def(X, N, D, Y), repl_defs(Y, T, RX).


% Remplace les expressions complexes d'une Abox et met toutes les expressions sous forme normale négative.
% Paramètres:
%   - Abox des assertions de concept non traitée
%   - Tbox
%   - Abox des assertions de concept traitée

traitement_Abox([], _, []).
traitement_Abox([(N, A) | T], D, [(N, RA) | RT]) :- traitement_Abox(T, D, RT), repl_defs(A, D, B), nnf(B, RA).


% Exécute la première étape de l'algorithme.

premiere_etape(Tbox, Abi, Abr) :-
  % check_Tbox(Tbox1),
  % check_Abi(Abi1),
  % check_Abr(Abr),
  init_Tbox(Tbox1),
  init_Abi(Abi1),
  init_Abr(Abr),
  traitement_Tbox(Tbox1, Tbox),
  traitement_Abox(Abi1, Tbox, Abi).


% Partage la liste d'assertions de concept de la Abox en cinq listes en fonction du type d'assertion.
% Exemples:
%   tri_Abox([(a, some(r1, c1))], Lie, Lpt, Li, Lu, Ls)
%     -> Lie = [(a, some(r1, c1))], Lpt = Li = Lu = Ls = []
%   tri_Abox([(a, some(r1, c1)), (b, all(r2, c2))], Lie, Lpt, Li, Lu, Ls)
%     -> Lie = [(a, some(r1, c1))], Lpt = [(b, all(r2, c2))], Li = Lu = Ls = []

tri_Abox([], [], [], [], [], []).
tri_Abox([(I, some(R, C)) | Abi], [(I, some(R, C)) | Lie], Lpt, Li, Lu, Ls) :- tri_Abox(Abi, Lie, Lpt, Li, Lu, Ls), !.
tri_Abox([(I, all(R, C)) | Abi], Lie, [(I, all(R, C)) | Lpt], Li, Lu, Ls) :- tri_Abox(Abi, Lie, Lpt, Li, Lu, Ls), !.
tri_Abox([(I, and(C1, C2)) | Abi], Lie, Lpt, [(I, and(C1, C2)) | Li], Lu, Ls) :- tri_Abox(Abi, Lie, Lpt, Li, Lu, Ls), !.
tri_Abox([(I, or(C1, C2)) | Abi], Lie, Lpt, Li, [(I, or(C1, C2)) | Lu], Ls) :- tri_Abox(Abi, Lie, Lpt, Li, Lu, Ls), !.
tri_Abox([(I, X) | Abi], Lie, Lpt, Li, Lu, [(I, X) | Ls]) :- tri_Abox(Abi, Lie, Lpt, Li, Lu, Ls).


% Transforme toutes les assertions de type 'i : ∃ R.C' en assertions 's : C' et 'i, s : R'
% Exemples:
%   complete_some([(a, some(r, c))], Lpt, Li, Lu, Ls, Abr)
%     -> Ls = [(inst1, c) | _], Abr = [(a, inst1, r) | _]

complete_some([], _, _, _, _, _).
complete_some([(I, some(R, C)) | Lie], Lpt, Li, Lu, [(S, C) | Ls], [(I, S, R) | Abr]) :- complete_some(Lie, Lpt, Li, Lu, Ls, Abr), genere(S).


% Code pour la génération de noms

compteur(1).

concat2([],L1,L1).
concat2([X|Y],L1,[X|L2]) :- concat2(Y,L1,L2).

genere(Nom) :- compteur(V),nombre(V,L1), concat2([105,110,115,116],L1,L2), V1 is V+1, dynamic(compteur/1), retract(compteur(V)), dynamic(compteur/1), assert(compteur(V1)), name(Nom,L2), !.
nombre(0,[]).
nombre(X,L1) :- R is (X mod 10), Q is ((X-R)//10), chiffre_car(R,R1), char_code(R1,R2), nombre(Q,L), concat2(L,[R2],L1).
chiffre_car(0,'0').
chiffre_car(1,'1').
chiffre_car(2,'2').
chiffre_car(3,'3').
chiffre_car(4,'4').
chiffre_car(5,'5').
chiffre_car(6,'6').
chiffre_car(7,'7').
chiffre_car(8,'8').
chiffre_car(9,'9').
