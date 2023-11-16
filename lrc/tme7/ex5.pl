% Exercice 5

et(1, 1, 1).
et(1, 0, 0).
et(0, 1, 0).
et(0, 0, 0).

ou(1, 1, 1).
ou(1, 0, 1).
ou(0, 1, 1).
ou(0, 0, 0).

non(0, 1).
non(1, 0).

xor(X, Y, Z) :- et(X, B, C), et(Y, A, D), ou(C, D, Z), non(X, A), non(Y, B).
circuit(X, Y, Z) :- non(X, A), et(X, Y, B), non(B, C), xor(A, C, D), non(D, Z).
