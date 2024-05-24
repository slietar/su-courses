#show raw.where(block: true): txt => rect(width: 100%, stroke: gray)[#txt]
#set text(11pt, lang: "fr")
// #show outline.where(): set text(size: 6pt)
#show heading.where(level: 1): it => [
  #set text(size: 24pt)
  #pad(y: 2pt, it.body)
]
#show heading.where(level: 2): set text(size: 18pt)
#show heading.where(level: 2): it => pad(y: 6pt, it.body)
#show heading.where(level: 3): set text(size: 14pt)
#show heading.where(level: 3): it => pad(y: 4pt, it.body)
#show outline: set heading(level: 2)
#show heading: set text(font: "IBM Plex Sans")
#show raw: set text(font: "Menlo")
#show link: underline

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

#align(center)[
  #v(15em)
  == Projet Bio-Informatique et Modélisation
  = Bases moléculaires du syndrome de Marfan, focus sur la protéine géante fibrilline 1

  #v(4em)

  #set text(size: 16pt)

  Zaina Dali

  Simon Liétar

  #v(4em)

  _sous la supervision de_

  Louis Carrel-Billiard

  Elodie Laine

  #v(4em)

  Mai 2024
]


#pagebreak()

#outline(indent: 2em)



#pagebreak()


== Introduction

Le syndrome de Marfan est une maladie génétique relativement rare présente chez l'Homme et qui affecte le tissu conjonctif. Elle est caractérisée par une variété de manifestations cliniques, notamment des anomalies cardiaques, ophtalmologiques et squelettiques. Bien que des progrès aient été réalisés dans la compréhension et la prise en charge de cette maladie, des défis persistent, avec notamment une morbidité substantielle et une mortalité prématurée qui demeure. Au niveau moléculaire, le syndrome de Marfan est principalement causé par des mutations dans le gène _FBN1_ codant pour la protéine fibrilline~1, une composante essentielle de la matrice extracellulaire.

Ce rapport se concentre sur l'étude des bases moléculaires du syndrome de Marfan, en mettant l'accent sur la protéine géante fibrilline~1. Notre objectif est d'établir une classification structurale, évolutive et fonctionnelle des mutations faux sens observées dans la fibrilline~1. Cette classification revêt une importance capitale pour mieux comprendre la relation entre le génotype et le phénotype associés au syndrome de Marfan, ainsi que pour explorer les implications cliniques de ces mutations.

Le travail sur ce projet s'est fait en collaboration avec Pauline Arnaud, Nadine Hannah et Laurent Gouya, praticiens de l'hôpital Bichat qui travaillent sur la compréhension du syndrome de Marfan et sont à l'origine du projet. Ceux-ci nous ont fourni une liste de phénotypes associés à des mutations observées chez des patients.

La première phase du projet a consisté en l'extraction de descripteurs pouvant servir à la classification des mutations. La seconde phase du projet a eu pour but de comprendre les relations entre ces descripteurs pour arriver à une classification non supervisée, et de mettre celle-ci en parallèle avec les phénotypes connus.

En combinant des approches bio-informatiques, structurales et évolutives, ce rapport propose une démarche multidimensionnelle pour cartographier les mutations de la fibrilline~1.


== Présentation de la fibrilline 1

La protéine fibrilline 1 est une protéine de 2871 résidus qui agit comme composante majeure de la matrice extracellulaire (ECM). Cette protéine est codée par le gène _FBN1_ de 257~kbp localisé sur le chromosome 15 @noauthor_gene_nodate @noauthor_p35555_nodate.

Malgré sa taille importante, cette protéine est essentiellement composée de domaines semblables @hubmacher_genetic_2011 :

- 4 domaines *EGF* semblables à l'epidermal growth factor qui contiennent chacun 6 cystéines.
- 43 domaines *EGFCB* (CB for calcium-binding) semblables à l'epidermal growth factor, mais capables de contenir un atome de calcium à leur extrémité N-terminale. Ces domaines contiennent également 6 cystéines chacun et sont très abondants dans les protéines structurales de l'ECM.
- 9 domaines *TB* pour TGFβ-binding protein-like. Ces domaines contiennent chacun 8 cystéines.

Nous utilisons ici sur les annotations d'UniProt, mais la classification de ces domaines peut varier d'une base de données à l'autre. La répartition des domaines dans la protéine est donnée dans les figures suivantes dont la @plddt.

La fibrilline 1 a de nombreux partenaires d'intéraction tels que les protéines MFAP, la fibrilline 2 ou encore la fibrilline 1 elle-même @hubmacher_fibrillins_2006. L'étude de ces intéractions permettrait sûrement d'en savoir plus sur son fonctionnement, mais nous avons choisi de concentrer notre attention sur la protéine seule.

La présence de ces partenaires d'intéractions laisse penser que les domaines semblables ont des rôles différents malgré leur apparente similarité. Nous avons donc essayé d'exploiter les comparaisons entre domaines pour comprendre le rôle de leurs différences.



== Structure de la fibrilline 1

Il n'existe pas de structure expérimentale complète de la fibrilline 1, mais seulement des structures de sous-ensembles comprenenant quelques domaines @noauthor_p35555_nodate. Nous avons utilisé AlphaFold 2 @jumper_highly_2021 puis 3 @abramson_accurate_2024 pour obtenir une structure de la protéine. Bien que la mesure de confiance locale de la prédiction (pLDDT) soit correcte avec AlphaFold 3 (entre 70 et 90 sur 100 pour la majorité de la protéine, voir @plddt), l'erreur de la position par rapport aux autres résidues (PAE) est trop importante (30~Å, voir @pae_alphafold3) pour pouvoir conclure sur la forme générale de la protéine et sur les intéractions entre les domaines. En particulier, AlphaFold a tendance à produire une structure globulaire, là où la littérature donne plus de crédit à une structure linéaire et rigide @hubmacher_fibrillins_2006.

Les régions hors des domaines EGF, EGFCB et TB ont toujours un pLDDT très faible (\~~20), probablement en raison de leur absence des protéines dont la structure expérimentale est connue.

En raison de la faible fiabilité des prédictions de la protéine entière, nous avons utilisé AlphaFold 2 pour prédire la structure des domaines individuellement, en incluant également les deux domaines adjacents afin d'avoir un «~contexte~» qui améliore la prédiction. En conséquence, les régions entre des domaines ne sont pas prédites ni analysées dans le reste du projet. Le pLDDT est meilleur que les prédictions de la protéine entière et le PAE assez bon au sein de chaque domaine (voir @pae). Nous avons également testé ESMFold @rives_biological_2021 mais obtenu des résultats moins bon qu'avec AlphaFold.

#figure(
  image("../output/plddt.png", width: 18cm),
  caption: [
    pLDDT pour différentes prédictions
  ],
) <plddt>

Nous avons inclus le pLDDT dans les descripteurs car celui-ci peut servir comme reflet de la structure secondaire de la protéine.

#figure(
  image("../output/pae_alphafold3.png", width: 18cm),
  caption: [
    Erreur d'alignment prédite (PAE) sur la prédiction d'AlphaFold 3 pour la protéine complète
  ],
) <pae_alphafold3>

#figure(
  image("../output/pae.png", width: 18cm),
  caption: [
    Erreur d'alignment prédite (PAE) sur les prédictions d'AlphaFold 2 pour les domaines individuels et les domaines adjacents
  ],
) <pae>


== Mutations

De nombreuses mutations de _FBN1_ sont connues comme responsables du syndrome de Marfan, et une mutation sur une seule des deux copies du gène est suffisante pour le provoquer. Le phénotype d'une mutation est toutefois très variable.

Les médecins de l'hôpital Bichat nous ont fourni une liste de 731 mutations faux sens observées sur des patients et comportant chacune zéro ou plus de 6 phénotypes, comme suit :

#figure(
  table(
    align: (left, center),
    columns: 2,
    stroke: none,
    table.header[*Effet*][*Nombre de mutations*],
    table.hline(stroke: gray),
    [Pneumothorax], [8],
    [Problème cardiaque\*], [24/486],
    [Problème cutané], [21],
    [Problème ophtalmologique\*], [16/434],
    [Problème neurologique], [6],
    [Problème squelettique\*], [26/485],
    table.hline(stroke: gray),
    [Problème grave], [46],
  ),

  caption: [Effets recensés des mutations dans _FBN1_. Les effets marqués d'une astérisque (\*) sont donnés avec deux niveaux de confiance ; la seconde valeur correspond au niveau le plus élevé.]
)

Dans la suite du suite du projet, on considèrera comme pathogènes les mutations ayant au moins un effet avec le niveau de confiance le plus élevé, le cas échéant. Cela concerne 584 mutations sur 421 résidus.

#figure(
  image("../output/mutations.png", width: 18cm),
  caption: [
    Positions avec des mutations pathogènes. Les traits bleus représentent de mutations pathogènes et rouges pathogènes et graves. Les surfaces en gris sont les régions d'intérêt pour les médecins de l'hôpital Bichat.
  ],
) <mutations>

Les mutations sont distribuées comme montré à la @mutations. Celle-ci sont assez uniformément distribuées, sauf pour les régions entre les domaines où elles sont moins nombreuses.

Les médecins de l'hôpital Bichat ont déterminé deux régions comme particulièrement importantes. La première, dite région néonatale, contient de nombreuses mutations graves qui sont responsables de troubles dès la naissance. La seconde, dite TB 5 (TB 7 dans les annotations UniProt que nous utilisons), n'a pas de signature spécifique sur la @mutations, mais est caractéristique de symptômes particuliers du syndrome de Marfan, la dysplasie géléophysique ou acromicrique.


== Descripteurs structuraux


=== Surface accessible au solvant (SASA)

Il est probable que les mutations pusisent être classifiées selon qu'elles soient à la surface de la protéine, donc responsable d'intéractions, ou bien enfouies et donc affectant la structure interne de la protéine. Pour mesurer quantitativement cette propriété, nous avons utilisé la surface accessible au solvant (solvent-accessible surface area, SASA) @lee_interpretation_1971.

Pour ce faire, nous avons employé la bibliothèque FreeSASA @mitternacht_freesasa_nodate qui implémente l'algorithme de Shrake–Rupley @shrake_environment_1973. Celui-ci consiste à faire « rouler » une boule contre la surface de Van der Waals des atomes de la protéine. L'algorithme retourne, pour chaque résidu, un ratio entre la surface accessible au solvant dans cette structure et la surface théorique maximale qui est une fonction du type d'acide aminé.


=== Variance circulaire

Nous avons utilisé la variance circulaire comme une autre métrique pour caractériser l'emplacement d'un résidu dans la protéine. La variance circulaire de l'atome $a$ est un nombre entre 0 et 1 défini comme :

$
  "CV"_a = 1 - 1/N ||sum_(
    i != a\
    ||arrow(x)_i - arrow(x)_a||<c
  )^N (arrow(x)_i - arrow(x)_a)/(||arrow(x)_i - arrow(x)_a||)||
$

où $arrow(x)_i$ est la position de l'atome $i$.

La variance circulaire d'un résidu est définie comme la moyenne des variances circulaires de ses atomes.

Le seuil de distance $c$ (en Å) permet de contrôler quels atomes sont considérés comme faisant partie du voisinage de l'atome $a$. Pour $c = infinity$, tous les atomes sont considérés comme faisant partie du voisinage et $"CV"_a$ donne une mesure de l'enfouissement dans la protéine entière.

L'interprétation de la variance circulaire est la suivante: si celle-ci est à 1 comme en rouge dans la @cv-global, alors tous les vecteurs $arrow(x)_i - arrow(x)_a$ se sont annulés et l'atome est donc au centre de son voisinage. Si celle-ci est à 0, tous les vecteurs vont dans la même direction est l'atome est à la surface de la protéine.

#figure(
  image("../output/cv_global.png", width: 18cm),
  caption: [
    Fibrilline 1 colorée avec la variance circulaire avec $c = infinity$. Rouge, au centre: 1.0, bleu foncé, en surface: 0.0. La structure ne sert que comme illustration car elle n'est pas fiable (voir @plddt).
  ],
) <cv-global>

L'interprétation exacte dépend du seuil choisi. Un seuil infini n'est pas souhaitable dans notre cas car la structure de la protéine entière n'est pas fiable. En revanche, des seuils à 10 et 20 Å permettent de capturer le contraste de l'enfouissement des résidus à l'échelle de la structure secondaire ou d'un domaine.

Pour implémenter cet algorithme efficacement sur une grande protéine, nous avons écrit un shader en WGSL qui calcule la variance circulaire de tous les atomes en même temps sur le GPU. Nous avons publié #link("https://github.com/slietar/molcv")[un paquet Python et un paquet Rust] pour utiliser l'algorithme sur n'importe quelle protéine en ligne de commande ou programmatiquement. D'autres algorithmes pourraient être utilisés, par exemple une grille qui permette de rapidement déterminer quels atomes sont dans le voisinage de l'atome considéré.

#figure(
  image("../output/cv.png", width: 18cm),
  caption: [
    Variance circulaire pour différents seuils de distance donnés sur l'axe vertical
  ],
) <cv>

On observe à la @cv que la variance circulaire tend à se stabiliser à mesure que de plus en plus d'atomes sont inclus, jusqu'à ce que le seuil de distance atteigne la plus grande distance entre deux atomes.

#figure(
  grid(
    columns: 2,
    gutter: 2mm,
    image("../output/cv10.png"),
    image("../output/cv30.png")
  ),
  caption: [
    Structure de 3 domaines EGFCB adjacents colorés avec la variance circulaire, avec des seuils de 10~Å (gauche) et 30~Å (droite). Les régions en vert sont les domaines adjacents qui sont considérés lors du calcul du domaine central mais dont la variance circulaire n'est pas calculée.
  ]
)


=== Flexibilité et écart à la structure consensus

La présence de nombreuses répétitions des mêmes domaines donne la possibilité de comparer leurs structures. En alignant les séquences des domaines d'un type donné, puis leurs structures, on peut estimer la position «~consensus~» $arrow(x)_i$ de chaque résidu $i$ pour un type de domaine $k$ :

$
  arrow(x)_(i) = 1/N sum_(
    d in cal(A)_k
  )^(N) arrow(x)_(d,i)
$

où $arrow(x)_(d,i)$ correspond à la position moyenne (ou alternativement à la position de l'atome de carbone #sym.alpha) des atomes du résidu $i$ du domaine $d$, et $cal(A)_k$ à l'ensemble des domaines de type $k$.

On peut maintenant définir un nouveau descripteur pour chaque résidu comme l'écart à la position consensus correspondante. L'idée est que si l'un des domaines a obtenu une nouvelle fonction suite à mutation, celle-ci peut se refléter dans sa structure qui s'est alors éloignée de la position consensus.

On peut également calculer la distance moyenne de chaque domaine à la position consensus, pour chaque résidu $i$ d'un type de domaine, ce qui donne la flexibilité $F_i$ :

$
  F_i = 1/N sum_(
    d in cal(A)_k
  )^(N) ||arrow(x)_(d,i) - arrow(x)_i||
$


#figure(
  grid(
    columns: 2,
    gutter: 2mm,
    image("../output/flex_egfcb.png"),
    image("../output/flex_tb.png"),
  ),
  caption: [
    Structures de domaines EGFCB (gauche) et TB (droite) colorés avec la flexibilité. Gauche : bleu foncé 0.37~Å, rouge 3.80~Å. Droite : bleu foncé 0.53~Å, rouge 7.26~Å.
  ],
) <flex>

On observe à la figure @flex que les hélices #sym.alpha et feuillets #sym.beta sont les parties les moins flexibles des domaines, ce qui est attendu car ces structures secondaires sont plus sensibles aux changements de structure. Une exception notable est l'hélice #sym.alpha C-terminale des domaines TB qui est montrée comme flexible en raison de prédictions très éloignées dans quelques domaines.


=== DSSP

DSSP (Dictionary of Secondary Structure of Protein) @kabsch_dictionary_1983 est un algorithme standard largement utilisé pour attribuer la structure secondaire aux acides aminés d’une protéine à partir de ses coordonnées à résolution atomique. Pour cela, DSSP utilise un dictionnaire de liaisons hydrogène et des caractéristiques géométriques pour attribuer la structure secondaire en analysant les liaisons hydrogène du squelette protéique ainsi que la topologie des feuillets β pour chaque résidu.
Le résultat obtenu est un code unique attribué à chaque résidu, indiquant sa structure secondaire, comme par exemple H pour une α-hélice, B pour un pont β isolé, E pour un brin étendu, et ainsi de suite.
Nous avons appliqué l'algorithme DSSP @minami_pydssp_nodate à la fibrilline 1 et avons obtenu les résultats présentés dans la figure suivante.

#figure(
  image("../output/dssp.png", width: 18cm),
  caption: [ Le pourcentage de résidus mutés présents dans chaque structure secondaire de la fibrilline 1
  ],
)

Comme le montre la figure, nous remarquons que la structure majoritaire de notre protéine est la structure en boucle (LOOP), qui contient 43,09 % des résidus mutés présents dans notre protéine. Ensuite, nous observons la structure en feuillet bêta (BETA STRAND), qui est moins fréquente que la boucle, mais qui représente 47,2 % des résidus mutés. Enfin, la structure la moins présente est celle avec 9,71 % de résidus mutés.



== Descripteurs non-structuraux


=== GEMME

Nous avons utilisé l'algorithme GEMME @laine_gemme_2019 pour obtenir un descripteur qui soit représentatif de la pression évolutive de chaque mutation. Pour ce faire, GEMME se fonde sur un alignment multi-séquence de la séquence d'intérêt et construit un modèle évolutif pour prédire l'effet de chaque acide aminé à chaque position.

Le résultat de GEMME étant donné sous forme d'une matrice avec le score pour chaque acide aminé pour chaque résidu, nous avons réduit cette matrice en deux descripteurs : le score GEMME moyen à chaque position et le score GEMME de la pire mutation à chaque position. Un score élevé indique une pression évolutive faible.

#figure(
  image("../output/gemme.png", width: 18cm),
  caption: [
    Scores GEMME pour différentes mutations
  ],
) <gemme>

Nous avons aussi calculé le score GEMME en ne considérant que les séquences annotées comme orthologues dans 156 espèces et donc assurément liées évolutivement à _FBN1_. En soustrayant les deux scores, on obtient une mesure différence entre les distances évolutives d'une mutation selon les deux contextes, que l'on utilisera comme descripteur noté #sym.Delta GEMME. Une valeur élevée correspond à une pression de sélection faible dans le contexte complet et élevée dans le contexte resteint limité aux orthologues.

La @gemme montre que les mutations sur lesquelles nous nous basons ont généralement un score faible ($< 2$) voire  très faible, quel que soit le nombre d'effets recensé. Les mutations recensées sur gnomAD, dont peu ont un phénotype connu, ont également tendance à avoir un score faible lorsque qu'elle sont annotées comme pathogènes.

L'outil PRESCOTT @tekpinar_prescott_2024 se base sur GEMME et améliore les prédictions en prenant en compte les informations structurales provenant d'AlphaFold et les fréquences alléliques provenant de gnomAD @chen_genomic_2024. Nous n'avons pas utilisé PRESCOTT en raison de la faible fiabilité des prédictions AlphaFold pour la fibrilline 1.


=== Score de polymorphisme

Afin de représenter la diversité des mutations connues, nous avons ajouté un « score de polymorphisme » qui est dérivé, pour chaque résidu, du nombre de mutations faux sens observées dans la base de données gnomAD @chen_genomic_2024. Le but étant de mesurer quels résidus sont abondants sans être dangereux, nous n'avons pas compté les résidus annotés comme « pathogènes », « pathogènes/probablement pathogènes » ou « probablement pathogènes ».

#figure(
  image("../output/polymorphism_histogram.png", width: 18cm),
  caption: [
    Histogramme des observations
  ],
)

Nous avons utilisé une échelle logarithmique pour pallier aux fréquences d'apparitions distribuées inégalement.


== Interprétation


=== Corrélations entre descripteurs

#figure(
  image("../output/residues_correlations.png", width: 18cm),
  caption: [
    Corrélations entre descripteurs
  ],
) <corr>

L'étude des corrélations à la @corr montre de fortes corrélations entre les descripteurs les variances circulaires à 10 et 20~Å (+0.80), ainsi que le score GEMME et le descripteur #sym.Delta GEMME calculé à partir des scores GEMME (+0.80). On observe également une corrélation importante entre la variance circulaire avec un faible seuil et la surface accessible au solvant (–0.91). Ceci est attendu car ces deux variables mesurent des propriétés semblables des résidus bien qu'avec des méthodes différentes. La corrélation est négative car la SASA est élevée et la variance circulaire faible lorsqu'un résidu est en surface.

Une autre corrélation notable est la correspondance entre le score GEMME moyen et la SASA (+0.56). Ceci révèle que les résidus en surface ont tendance à avoir une faible pression de sélection. Enfin, on note une corrélation entre le score de polymorphisme et le score GEMME moyen (+0.30), qui s'explique facilement par la faible pression de sélection, donc score GEMME élevé, dans les positions polymorphiques.


=== Analyse en composantes principales

Nous avons effectué des analyses en composantes principales (PCA) pour réduire la dimensionnalité des données et en obtenir une visualisation. Plus précisement, nous avons effectué une PCA par type de domaine pour prendre en compte les différences possibles entre eux. La PCA ne prend en compte que les positions pathogènes, mais toutes les positions sont affichées dans la @pca. Les données sont mises à l'échelle avant la PCA. Nous n'avons pas inclus de variables discrètes, tel que la structure secondaire, car celles-ci donnent des résultats peu interprétables dans une PCA.

#figure(
  image("../output/residues_pca.png", width: 18cm),
  caption: [
    Composantes PC1 et PC2 de PCA par type de domaine. Les points rouges correspondent à des positions pathogènes.
  ]
) <pca>

Il n'y pas de clusters évidents mais on observe une nette tendance pour les positions pathogènes à se trouver dans un sous-ensemble de l'espace.

#figure(
  image("../output/residues_pca_components.png", width: 18cm),
  caption: [
    Contribution des descripteurs aux composantes PC1 et PC2
  ]
) <pca_comp>

Les contributions des descripteurs sont semblables d'un type de domaine à l'autre, mis à part quelques différences. Le premier mode (PC1), avec \~~45 % de variance expliquée, donne un poids important à la pression de sélection (GEMME, polymorphisme) et à l'exposition (SASA, variance circulaire), et regroupe les descripteurs corrélés, tels que les variances circulaires.

Une valeur élevée de PC1 correspond à une position en surface avec une faible pression de sélection. La plupart des positions ayant une valeur faible de PC1 (et plus faible que la moyenne des positions), on conclut que celles-ci sont généralement enfouies et avec une forte pression de sélection. Les domaines EGF ont également une forte contribution du pLDDT dont la signification n'est pas facilement interprétable.

Le mode PC2 explique \~ 22~% de la variance. Une valeur élevée de PC2 dans les domaines EGF et TB, ou faible dans les domaines EGFCB, correspond à une position en surface mais avec une pression de sélection élevée. Ce mode donne aussi une contribution non négligeable à l'écart avec la structure consensus selon les types de domaines. Il est toutefois difficile de conclure sur une classification à partir de cette composante.

#figure(
  image("../output/pca_map.png", width: 18cm),
  caption: [
    Composantes PC1 et PC2 des mutations des domaines EGFCB, localisées dans la protéine
  ]
) <pca_map>

L'étude de la distribution des valeurs PC1 et PC2 dans les positions pathogènes des domaines EGFCB de la protéine (@pca_map) montre que PC1 est généralement faible sauf pour quelques valeurs extrêmes, dans les domaines 30 et 32 par exemple. Pour PC2, les valeurs sont plus uniformes au sein d'un domaine, avec des domaines avec une valeur souvent faible de PC2 (par exemple 19 et 29), et d'autres une valeur élevée (par exemple 9, 10 et 46). Une analyse plus poussée sur les positions des mutations et la structure correspondante serait nécessaire pour conclure.


=== Classification en fonction de la pression de sélection et de l'exposition

Nous proposons une classification plus simple, sur la base de la pression de sélection (score GEMME moyen) et de l'exposition (variance circulaire), avec les classes suivantes :

- résidus enfouis (variance circulaire $> 0.5$) avec une forte pression de sélection (GEMME $< -2.5$), classe qui comprend la grande majorité des mutations (en bleu dans la @gemme_cv) ;
- résidus enfouis avec une faible pression de sélection (en violet) ;
- résidus en surface avec une faible pression de sélection (en jaune).

Les deux dernières classes contiennent chacune une vingtaine de mutations, et sont intéressantes parce qu'elles contrastent avec le reste des positions.

#figure(
  image("../output/gemme_cv.png", width: 18cm),
  caption: [
    Classification possible des mutations
  ]
) <gemme_cv>


== Conclusion

Nous avons prédit la structure de la fibrilline 1 et évalué la fiabilité de cette prédiction. Sur cette base, nous avons dérivé des descripteurs structuraux pour chaque position, ainsi que des descripteurs non-structuraux.

En utilisant ces descripteurs, nous avons tenté d'effectuer une classification non supervisée des résidus de la protéine en tenant contenu des positions ayant des mutations connues comme pathogènes, telles que décrites par nos collaborateurs à l'hôpital Bichat. Nous n'avons pas pu mettre en évidence de groupes bien définis de résidus à partir des descripteurs. Il est probable que différents mécanismes moléculaires soient responsables de la pathogénicité des mutations, visible notamment par l'étendue de l'exposition des résidus, la large répartition des mutations dans la protéine, ainsi que la diversité de phénotypes. La présence de mutations à des positions présentant une faible pression de sélection nous interroge aussi sur le fonctionnement de ces mutations.

Parmi les extensions possibles au projet, nous pouvons par exemple citer l'étude des épissages alternatifs ou encore l'étude des partenaires et de leur position de liaison avec la fibrilline 1.


#pagebreak()
#bibliography("MFS.bib")
