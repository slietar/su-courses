% TME8
% Simon LIÉTAR


% Exercice 2

% Le prédicat est vrai si la première liste est vide et que la seconde est identique au résultat.
concatene([], A, A).

% On retire les premiers éléments A de la première liste et du résultat en vérifiant qu'ils sont identiques, puis en répétant l'opération avec ces deux listes raccourcies, B et D. La seconde liste ne change pas.
concatene([A | B], C, [A | D]) :- concatene(B, C, D).


% L'inverse d'une liste vide est une liste vide.
inverse([], []).

% On retire le premier élément A de la première liste qui devient ainsi B, puis on essaie de retrouver la seconde liste C en concaténant une liste inconnue D à A, pour vérifier que le dernier élement de C est A. Enfin on répète l'opération sur les listes B et D qui sont chacune raccourcie à une extrémité différente.
inverse([A | B], C) :- concatene(D, [A], C), inverse(B, D).


% La suppression de n'importe quel élément d'une liste vide donne une liste vide.
supprime([], _, []).

% Si le premier élément A de la liste en entrée est différent de l'élément à supprimer, on s'attend à ce que ce soit aussi le premier élément du résultat, et on appelle le prédicat à nouveau sur les liste d'entrée et de résultat raccourcies.
supprime([A | B], C, [A | D]) :- supprime(B, C, D), A \== C.

% Si le premier élément A de la liste en entrée est identique à l'élément à supprimer, on appelle le prédicat à nouveau sur la liste raccourcie et avec le même résultat attendu, c'est-à-dire sans cet élément.
supprime([A | B], A, C) :- supprime(B, A, C).


% Un filtre vide donne la même liste qu'en entrée.
filtre(A, [], A).

% On retirer le premier élement B de la liste filtre, puis on produit une nouvelle liste E où B a été supprimé de la liste d'entrée A, et répète l'opération sur la liste E et la liste filtre raccourcie.
filtre(A, [B | C], D) :- supprime(A, B, E), filtre(E, C, D).


% Exercice 3

% On vérifie que la liste A est identique à son inverse.
palindrome(A) :- inverse(A, A).

% Une liste vide est une palindrome.
palindrome2([]).

% Une liste ne contenant qu'un seul élément est un palindrome.
palindrome2([_]).

% On retire le premier élément A de la liste d'entrée, puis on essaie de produire une nouvelle liste C et concaténant A à celle-ci pour obtenir la liste d'entrée raccourcie, afin de vérifier que le dernier élément est A. Enfin on répète l'opération sur la liste C.
palindrome2([A | B]) :- concatene(C, [A], B), palindrome2(C).
