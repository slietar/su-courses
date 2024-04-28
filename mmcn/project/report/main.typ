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

La nullcline $x' = 0$ est une courbe cubique et la nullcline $y' = 0$ est une parabole.

// ce qui donne $x^3 + 2x^2 + (-z-1) = 0$.

// Pour $z = -1$, on a $x^3 + 2x^2 = 0$ soit $x = 0$ ou $x = -2$.

=== Question (b)

La jacobienne en un point stationnaire $(v_0, w_0)$ est donnée par :

$ J(v_0, w_0) = mat(-3v_0^2 + 6v_0, 1; -10v_0, -1) $

La condition pour avoir des valeurs propres complexes conjuguées, c'est-à-dire un foyer ou un centre, est donnée par $Delta = tr J(v_0, w_0)^2 - 4det J(v_0, w_0) < 0$, c'est-à-dire $0.0371 < v_0 < 3.2681$.

// Les points stationnaires sont à l'intersection des deux nullclines, donc l'équation $x^3 + 2x^2 - z - 1 = 0$.

// $ lambda_1 + lambda_2 = tr J(v_0, w_0)  $
// $ lambda_1 lambda_2 = det J(v_0, w_0) = $

=== Question (c)

On trace le diagramme de bifurcation obtenu à partir de la solution analytique des points stationnaires.

#figure(
  image("../output/bifurcation.png"),
  caption: [Diagramme de bifurcation (sans le cycle limite)]
)

On observe quatre bifurcations, pour $z$ allant de -2 à 12 :

1. En $z = -1$, une bifurcation pli cause l'apparition de deux points fixes stables.
2. En $z approx -0.93$ ou $x = 1 - sqrt(6)/3 approx 0.18$, une bifurcation de Hopf cause l'apparition d'un cycle limite stable autour d'un point fixe instable précédemment stable.
3. En $z = 5/27 approx 0.19$, une bifurcation pli cause la disparition des deux points fixes stables.
4. En $z approx 11.59$ ou $x = 1 + sqrt(6)/3 approx 1.82$, une bifurcation de Hopf cause la disparition du cycle limite.

#figure(
  image("../output/eigenvalues.png"),
  caption: [Valeurs propres]
)
