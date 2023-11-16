% Exercice 4

pere(pepin, charlemagne).
pere(pepin, carloman).
pere(charlemagne, alpais).
pere(charlemagne, adelaide).
mere(berthe, charlemagne).
mere(berthe, carloman).
mere(himiltrude, alpais).
mere(hildegarde, adelaide).

parent(X, Y) :- pere(X, Y).
parent(X, Y) :- mere(X, Y).
parents(X, Y, Z) :- pere(X, Z), mere(Y, Z).
grandPere(X, Y) :- pere(X, Z), parent(Z, Y).
frereOuSoeur(X, Y) :- parent(Z, X), parent(Z, Y), X \== Y.
ancetre(X, Y) :- parent(X, Y).
ancetre(X, Y) :- parent(X, Z), ancetre(Z, Y).
