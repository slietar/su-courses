#show raw.where(block: true): txt => rect(width: 100%, stroke: gray)[#txt]
#set text(11pt)
#show heading.where(level: 1): set text(size: 18pt)
#show heading.where(level: 2): set text(size: 16pt)
#show heading.where(level: 2): it => pad(y: 6pt, it.body)
#show heading: set text(font: "Helvetica Now Display")
#show raw: set text(font: "Menlo")

#set page(
  margin: (x: 72pt, y: 72pt),
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
  caption: [Longitude Position des bars et restaurants dans Paris]
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
