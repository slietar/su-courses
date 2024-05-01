#show raw.where(block: true): txt => rect(width: 100%, stroke: gray)[#txt]
#set text(11pt)
#show heading.where(level: 1): set text(size: 18pt)
#show heading.where(level: 2): set text(size: 16pt)
#show heading.where(level: 2): it => pad(y: 6pt, it.body)
#show heading.where(level: 3): set text(size: 12pt)
#show heading: set text(font: "Helvetica Now Display")
#show raw: set text(font: "Menlo")

#show figure.caption: it => [
  #strong[
    #it.supplement
    #context it.counter.display(it.numbering)
  ]
  ~
  #context it.body
]

#set page(
  margin: (x: 72pt, y: 72pt),
  numbering: "1"
)


= Les modèles de Hindmarsh-Rose

== 1. Premier régime

=== Question (a)

Avec $c = 1$, le système devient :

$ cases(
  x' = y - x^3 + 3x^2 + z,
  y' = 1 - 5x^2 - y
) $

En posant $x' = 0$ et $y' = 0$, on obtient les nullclines suivantes :

$ cases(
  y = x^3 - 3x^2 - z #h(.5cm) & (x' = 0),
  y = 1 - 5x^2 & (y' = 0)
) $

La nullcline $x' = 0$ est une courbe cubique et la nullcline $y' = 0$ est une parabole. Un exemple est donné à la @trajectories.

// ce qui donne $x^3 + 2x^2 + (-z-1) = 0$.

// Pour $z = -1$, on a $x^3 + 2x^2 = 0$ soit $x = 0$ ou $x = -2$.

#figure(
  image("../output/trajectories.png"),
  caption: [Plan de phase avec $z = 3$]
) <trajectories>

=== Question (b)

La jacobienne en un point stationnaire $(v_0, w_0)$ est donnée par :

$ J(v_0, w_0) = mat(-3v_0^2 + 6v_0, 1; -10v_0, -1) $

La condition pour avoir des valeurs propres complexes conjuguées, c'est-à-dire un foyer ou un centre, est donnée par :

$
  Delta = tr J(v_0, w_0)^2 - 4det J(v_0, w_0) &< 0 \
  9v_0^4 - 36v_0^3 + 30v_0^2 - 28v_0 + 1 &< 0 \
  0.0371 < v_0 < 3.2681
$

La condition pour avoir une bifurcation col-nœud est le passage d'une valeur propre par zéro, à condition de ne pas avoir de valeurs propres conjuguées :

$
  det J(v_0, w_0) &= 0 \
  3v_0^2 + 4v_0 &= 0 \
  v_0 = -4/3 "ou" v_0 = 0
$

La condition pour avoir une bifurcation de Hopf est le passage de la partie réelle d'une paire de valeurs propres complexes conjuguées par zéro :

$
  tr J(v_0, w_0) &= 0 \
  -3v_0^2 + 6v_0 - 1 &= 0 \
  v_0 = 1 - sqrt(6)/3 "ou" v_0 = 1 + sqrt(6)/3
$

On vérifie ces critères en traçant les valeurs propres des points stationnaires à la @eigenvalues.

#figure(
  image("../output/eigenvalues.png"),
  caption: [Valeurs propres des points stationnaires]
) <eigenvalues>

=== Question (c)

Les points stationnaires sont les points d'intersection des deux nullclines et sont les solutions d'un polynôme de degré 3 que l'on peut donc résoudre analytiquement. On trace à la @bifurcation le diagramme de bifurcation obtenu à partir de ces solutions.

#figure(
  image("../output/bifurcation.png"),
  caption: [Diagramme de bifurcation]
) <bifurcation>

On observe cinq bifurcations pour $z$ allant de -2 à 12 :

1. En $z = -1$ et $x = 0$, une bifurcation col-nœud cause l'apparition d'un nœud stable et d'un col, en plus du nœud stable déjà existant.
2. En $z approx -0.93$ et $x = 1 - sqrt(6)/3 approx 0.18$, une bifurcation de Hopf transforme un foyer stable (précédemment un nœud jusqu'en $z approx -0.997$ et $x approx 0.037$) en un foyer instable, mais sans cycle limite en raison de la collision de celui-ci avec le col.
3. En $z approx -0.07$ et $x approx -0.92$, une bifurcation homocline cause l'apparition d'un cycle limite stable.
3. En $z = 5/27 approx 0.19$ et $x = -4/3$, une bifurcation col-nœud cause la disparition du col et du nœud restant.
4. En $z approx 11.59$ et $x = 1 + sqrt(6)/3 approx 1.82$, une bifurcation de Hopf cause la disparition du cycle limite et la transformation du foyer instable en un foyer stable.
