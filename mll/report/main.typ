#show raw.where(block: true): txt => rect(width: 100%, stroke: gray)[#txt]
#set text(11pt)
#show heading.where(level: 1): set text(size: 18pt)
#show heading.where(level: 2): set text(size: 16pt)
#show heading.where(level: 2): it => pad(y: 6pt, it.body)
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



= TME 1 – Estimation de densité


== Introduction

Dans la classe `Density`, on explicite le fait que la classe soit abstraite, on ajoute des signatures aux méthodes et on implémente la méthode ```py Density.score()```.

Le score est calculé comme $sum_(i) log(y_i)$. L'ajout de $10^(-10)$ aux prédictions permet d'éviter que les valeurs nulles ne causent un score infini.

```py
from abc import ABC, abstractmethod

class Density(ABC):
  @abstractmethod
  def fit(self, data: np.ndarray):
    ...

  @abstractmethod
  def predict(self, data: np.ndarray) -> np.ndarray:
    ...

  def score(self, data: np.ndarray):
    return np.log(np.maximum(self.predict(data), 1e-10)).sum()
```


== Données

#figure(
  image("../output/tme1/1.png"),
  caption: [Position des bars et restaurants dans Paris]
)


== Méthode par histogramme

On utilise #link("https://numpy.org/doc/stable/reference/generated/numpy.digitize.html")[```py np.digitize()```] pour trouver les bins auxquelles appartiennent les données.

```py
class Histogramme(Density):
  def __init__(self, steps: int = 10):
    super().__init__()

    self.steps = steps

    self.edges: Optional[list[np.ndarray]] = None
    self.hist: Optional[np.ndarray] = None

  def fit(self, x: np.ndarray):
    self.hist, self.edges = np.histogramdd(x, bins=self.steps, density=True)

  def predict(self, x: np.ndarray):
    assert self.edges is not None
    assert self.hist is not None

    return self.hist[*(np.digitize(x[:, dim], self.edges[dim][1:], right=True) for dim in range(x.shape[1]))]
```

#figure(
  image("../output/tme1/2.png"),
  caption: [Estimation de la densité de probabilité de bars, pour différents nombres de bins $N$]
)


On utilise un ensemble de test de 20 % des points et un ensemble d'entraînement avec les 80 % restants. La vraisemblance est maximale pour $N tilde.eq 10$ bins. Elle augmente lorsqu'il y a faible nombre de bins, puis diminue en raison du surentraînement.


#figure(
  image("../output/tme1/4.png"),
  caption: [Vraisemblance en fonction du nombre de bins $N$]
)


== Méthode à noyaux

On implémente les noyaux :

```py
def kernel_uniform(x: np.ndarray):
  return np.where(np.any(np.abs(x) <= 0.5, axis=-1), 1.0, 0.0)

def kernel_gaussian(x: np.ndarray, d: int = 2):
  return np.exp(-0.5 * (np.linalg.norm(x, axis=-1) ** 2)) / ((2 * np.pi) ** (d * 0.5))
```

Et la classe ```py KernelDensity``` :

```py
class KernelDensity(Density):
  def __init__(self, kernel: Optional[Callable[[np.ndarray], np.ndarray]], sigma: float = 0.1):
    super().__init__()

    self.kernel = kernel
    self.sigma = sigma
    self.x: Optional[np.ndarray] = None

  def fit(self, x: np.ndarray):
    self.x = x

  def predict(self, data: np.ndarray):
    assert self.kernel is not None
    assert self.x is not None

    return self.kernel((data[:, None, :] - self.x[None, :, :]) / self.sigma).sum(axis=1) / (self.sigma ** self.x.shape[1]) / self.x.shape[0]
```

#figure(
  image("../output/tme1/7.png"),
  caption: [Vraisemblance avec le noyau gaussien en fonction de $sigma$]
)

La vraisemblance est maximale avec le noyau gaussien sur les données de test est obtenue avec $sigma = 2.02 dot.op 10^(-3)$.


#figure(
  image("../output/tme1/8.png"),
  caption: [Vraisemblance avec le noyau uniforme en fonction de $sigma$]
)

La vraisemblance est maximale avec le noyau uniforme sur les données de test est obtenue avec $sigma = 9.83 dot.op 10^(-5)$.


== Régression par Nadaraya-Watson

#figure(
  image("../output/tme1/9.png"),
  caption: [Notes des bars dans Paris]
)

#figure(
  image("../output/tme1/10.png"),
  caption: [Erreur aux moindres carrés pour un noyau gaussien en fonction de $sigma$]
)

#figure(
  image("../output/tme1/11.png"),
  caption: [Erreur aux moindres carrés pour un noyau uniforme en fonction de $sigma$]
)

Les ensembles de test et d'entraînement n'ayant pas la même note moyenne, ils tendent vers des valeurs différentes lorsque $sigma arrow.r oo$. On observe que cette méthode n'est pas très efficace pour estimer la note d'un bar car l'erreur minimale n'est que très légèrement inférieur à la moyenne pour l'ensemble de test.



= TME 2 – Descente de gradient

== Expérimentations

=== Régression linéaire

#figure(
  image("../output/tme2/1.png"),
  caption: [Coût en fonction de l'itération]
)

#figure(
  image("../output/tme2/2.png"),
  caption: [Données et frontière de décision]
)

#figure(
  image("../output/tme2/3.png"),
  caption: [Profil de la fonction coût et trajectoire de la descente de gradient]
)


=== Régression logistique

#figure(
  image("../output/tme2/4.png"),
  caption: [Coût en fonction de l'itération]
)

#figure(
  image("../output/tme2/5.png"),
  caption: [Données et frontière de décision]
)

#figure(
  image("../output/tme2/6.png"),
  caption: [Profil de la fonction coût et trajectoire de la descente de gradient]
)

#figure(
  image("../output/tme2/7.png"),
  caption: [Profil de la fonction coût et trajectoire de la descente de gradient]
)



= TME 5 – Perceptron et SVMs

== Projections et pénalisation


=== Projection avec biais et projection polynomiale de degré 2

On code les fonction ```py proj_biais``` et ```py proj_poly``` :

```py
def proj_biais(x: np.ndarray, /):
  return np.c_[np.ones((*x.shape[:-1], 1)), x]
```

```py
import itertools

def proj_poly(x: np.ndarray, /):
  return np.c_[
    np.ones((*x.shape[:-1], 1)),
    x,
    *(x[..., a, None] * x[..., b, None] for a, b in itertools.combinations_with_replacement(range(x.shape[-1]), 2))
  ]
```

#figure(
  image("../output/tme5/5.png"),
  caption: [Séparation des données avec une projection avec biais et avec une projection polynomiale de degré 2]
)

L'ajout d'un biais uniquement ne permet pas de séparer les données car celles-ci ne sont pas linéairement séparables, or le modèle reste linéaire. En revanche, la projection polynomiale, et en particulier la composante $x_1 x_2$, permet de séparer les données. La frontière de décision du modèle pour les données de type 1 est de la forme $0.1 + 10x_1 x_2 = 0$.


=== Projection gaussienne

On code la fonction ```py proj_gauss``` :

```py
def proj_gauss(x: np.ndarray, /, base: np.ndarray, sigma: float):
  return np.exp(-0.5 * (np.linalg.norm(x[..., None, :] - base, axis=-1) / sigma) ** 2)
```

Pour les données de type 0, deux points de base bien placés suffisent pour séparer les données. Un seul point de base pourrait même suffire s'il on ajoute un biais.

#figure(
  image("../output/tme5/6.png"),
  caption: [Séparation des données de type 0 avec $sigma = 1.0$]
)

Si la distance des points de base avec les données ne permet pas de classer : [...]

#figure(
  image("../output/tme5/7.png"),
  caption: [Séparation des données de type 0 avec $sigma = 1.0$]
)

Pour les données de type 1 et 2, on peut créer une grille de points de base.

#figure(
  image("../output/tme5/9.png"),
  caption: [Séparation des données de type 1 avec $sigma = 1.5$]
)

Avec un $sigma$ trop élevé, les gaussiennes se chevauchent et il est impossible de séparer les données. Avec un $sigma$ trop faible, les gaussiennes deviennent négligeables dès que l'on s'éloigne des points de base.

#figure(
  image("../output/tme5/10.png"),
  caption: [Séparation des données de type 1 avec $sigma = 0.5$]
)

Le problème de l'échiquier peut être correctement résolu avec une grille de points de base si ceux si sont au moins aussi nombreux que les cases de l'échiquier. Les données sont plus finement intercalées, on utilise donc un sigma plus faible.

#figure(
  image("../output/tme5/11.png"),
  caption: [Séparation des données de type 2 avec $sigma = 0.5$]
)



= TME 6 – Bagging, boosting

== Forêts aléatoires

#figure(
  image("../output/tme6/3.png"),
  caption: [Erreur d'un modèle de forêts aléatoires sur les données de type 1. Chaque combinaison de paramètres est testée 5 fois et c'est l'erreur moyenne qui est rapportée ici.]
)

== Boosting : AdaBoost

Les données de type 0 sont linéairement séparables et le boosting ne n'a pas d'effet puisque la classification réussit en une étape. Sur les données de type 1 et 2, les arbres de décisions échouent à séparer les données car aucune unique frontière de décision ne donne de résultat satisfaisant : on sélectionne toujours la moitié des points d'une classe et la moitié des points de l'autre classe. Pour tester le boosting, on propose d'utiliser les données de type 1 en retirant les points dans l'un des quatre quadrants.

#figure(
  image("../output/tme6/1.png"),
  caption: [Classification sur les données de type 1. À gauche, les classifications individuelles avec la frontière de décision en gris. L'opacité correspond aux poids $D_t (i)$ utilisés par le classifieur. Les points en vert sont correctement classés et ceux en rouge mal classés. À droite, la classification en utilisant tous les classifieurs jusqu'à celui donné.]
)

#figure(
  image("../output/tme6/2.png"),
  caption: [Valeurs de $epsilon_t$ et $Z$ associées à la classification sur les données de type 1]
)

En ajoutant davantage plus que 3 classifieurs sur le même problème, on observe que l'erreur $Z$ continue de diminuer parce que la prédiction devient de plus en plus fiable bien que quasiment tous les points soient déjà bien classés. En revanche, la valeur de $epsilon_t$ ne diminue pas car les classifieurs individuels suivant ne sont pas meilleurs que les premiers étant donnée la distribution des poids.
