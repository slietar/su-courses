#show raw.where(block: true): txt => rect(width: 100%, stroke: gray)[#txt]
#set text(11pt)
#show heading.where(level: 1): set text(size: 18pt)
#show heading.where(level: 2): set text(size: 16pt)
#show heading.where(level: 2): it => pad(y: 6pt, it.body)
#show heading: set text(font: "Helvetica Now Display")
#show raw: set text(font: "Menlo")

#set page(
  margin: (x: 72pt, y: 72pt),
  numbering: "1"
)



= TME 1 – Estimation de densité


== Introduction

Dans la classe `Density`, on explicite le fait que la classe soit abstraite, on ajoute des signatures aux méthodes et on implémente la méthode ```py Density.score()```.

Le score est calculé comme $sum_(i) log(y_i)$. L'ajout de $10^(-10)$ aux prédictions permet d'éviter que les valeurs nulles ne causent un score infini.

```python
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
  image("../tme1/output/1.png"),
  caption: [Position des bars et restaurants dans Paris]
)


== Méthode par histogramme

On utilise #link("https://numpy.org/doc/stable/reference/generated/numpy.digitize.html")[```python np.digitize()```] pour trouver les bins auxquelles appartiennent les données.

```python
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
  image("../tme1/output/2.png"),
  caption: [Estimation de la densité de probabilité de bars, pour différents nombres de bins $N$]
)


On utilise un ensemble de test de 20 % des points et un ensemble d'entraînement avec les 80 % restants. La vraisemblance est maximale pour $N tilde.eq 10$ bins. Elle augmente lorsqu'il y a faible nombre de bins, puis diminue en raison du surentraînement.


#figure(
  image("../tme1/output/4.png"),
  caption: [Vraisemblance en fonction du nombre de bins $N$]
)


== Méthode à noyaux

On implémente les noyaux :

```python
def kernel_uniform(x: np.ndarray):
  return np.where(np.any(np.abs(x) <= 0.5, axis=-1), 1.0, 0.0)

def kernel_gaussian(x: np.ndarray, d: int = 2):
  return np.exp(-0.5 * (np.linalg.norm(x, axis=-1) ** 2)) / ((2 * np.pi) ** (d * 0.5))
```

Et la classe ```python KernelDensity``` :

```python
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
  image("../tme1/output/7.png"),
  caption: [Vraisemblance avec le noyau gaussien en fonction de $sigma$]
)

La vraisemblance est maximale avec le noyau gaussien sur les données de test est obtenue avec $sigma = 2.02 dot.op 10^(-3)$.


#figure(
  image("../tme1/output/8.png"),
  caption: [Vraisemblance avec le noyau uniforme en fonction de $sigma$]
)

La vraisemblance est maximale avec le noyau uniforme sur les données de test est obtenue avec $sigma = 9.83 dot.op 10^(-5)$.


== Régression par Nadaraya-Watson

#figure(
  image("../tme1/output/9.png"),
  caption: [Notes des bars dans Paris]
)

#figure(
  image("../tme1/output/10.png"),
  caption: [Erreur aux moindres carrés pour un noyau gaussien en fonction de $sigma$]
)

#figure(
  image("../tme1/output/11.png"),
  caption: [Erreur aux moindres carrés pour un noyau uniforme en fonction de $sigma$]
)

Les ensembles de test et d'entraînement n'ayant pas la même note moyenne, ils tendent vers des valeurs différentes lorsque $sigma arrow.r oo$. On observe que cette méthode n'est pas très efficace pour estimer la note d'un bar car l'erreur minimale n'est que très légèrement inférieur à la moyenne pour l'ensemble de test.



= TME 2 – Descente de gradient

== Expérimentations

=== Régression linéaire

#figure(
  image("../tme2/output/1.png"),
  caption: [Coût en fonction de l'itération]
)

#figure(
  image("../tme2/output/2.png"),
  caption: [Données et frontière de décision]
)

#figure(
  image("../tme2/output/3.png"),
  caption: [Profil de la fonction coût et trajectoire de la descente de gradient]
)


=== Régression logistique

#figure(
  image("../tme2/output/4.png"),
  caption: [Coût en fonction de l'itération]
)

#figure(
  image("../tme2/output/5.png"),
  caption: [Données et frontière de décision]
)

#figure(
  image("../tme2/output/6.png"),
  caption: [Profil de la fonction coût et trajectoire de la descente de gradient]
)
