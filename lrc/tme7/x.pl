% Exercice 1

% r(a, b).
% r(f(X), Y) :- p(X, Y).
% p(f(X), Y) :- r(X, Y).


% Exercice 2

% r(a, b).
% q(X, X).
% q(X, Z) :- r(X, Y), q(Y, Z).


% Exercice 3

% v révise
% s sérieux
% c est consciencieux
% d fait ses devoirs pour le lendemain
% r réussit

% v(X) :- s(X).
% d(X) :- c(X).
% r(X) :- v(X).
% s(X) :- d(X).
% c(p).
% c(z).
