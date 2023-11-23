
check_Ac([]).
check_Ac([(X, Y) | A]) :- iname(X), cnamea(Y), check_Ac(A).

check_Ar([]).
check_Ar([(X, Y, Z) | A]) :- iname(X), iname(Y), rname(Z), check_Ar(A).

check_T([]).
check_T([(X, Y) | A]) :- cnamena(X), concept(Y), check_T(A).

% check_Ac([(michelAnge,personne), (david,sculpture), (sonnets, livre), (vinci,personne), (joconde,objet)]).
% check_Ar([(michelAnge, david, aCree), (michelAnge, sonnets, aEcrit),(vinci, joconde, aCree)]).
% check_T([(sculpteur,and(personne,some(aCree,sculpture))), (auteur,and(personne,some(aEcrit,livre))), (editeur,and(personne,and(not(some(aEcrit,livre)),some(aEdite,livre)))), (parent,and(personne,some(aEnfant,anything)))]).


cnamea(anything).
cnamea(nothing).

concept(X) :- cnamea(X).
concept(X) :- cnamena(X).
concept(and(X, Y)) :- concept(X), concept(Y).
concept(or(X, Y)) :- concept(X), concept(Y).
concept(not(X)) :- concept(X).
concept(some(R, C)) :- rname(R), concept(C).
concept(all(R, C)) :- rname(R), concept(C).


pas-autoref(N, and(X, Y)) :- pas-autoref(N, X), pas-autoref(N, Y).
pas-autoref(N, or(X, Y)) :- pas-autoref(N, X), pas-autoref(N, Y).
pas-autoref(_, X) :- cnamea(X).
pas-autoref(N, X) :- cnamena(X), N \== X.
pas-autoref(N, not(X)) :- pas-autoref(N, X).
pas-autoref(N, some(_, C)) :- pas-autoref(N, C).
pas-autoref(N, all(_, C)) :- pas-autoref(N, C).


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


traitement_Tbox([], []).
traitement_Tbox([(N, A) | T], [(N, RA) | RT]) :- traitement_Tbox(T, RT), repl_equiv(A, B), nnf(B, RA).

repl_equiv(and(X, Y), and(RX, RY)) :- repl_equiv(X, RX), repl_equiv(Y, RY), !.
repl_equiv(or(X, Y), or(RX, RY)) :- repl_equiv(X, RX), repl_equiv(Y, RY), !.
repl_equiv(A, B) :- equiv(A, B), !.
repl_equiv(A, A).
