# RIPOST — audit empirique de confidentialité différentielle

Preuve de concept réalisée à la demande d'A. Imine (LORIA), à partir de
l'article *RIPOST: Two-Phase Private Decomposition for Multidimensional Data*
(Laouir & Imine, ESORICS 2025) et du code publié à
<https://github.com/AlaEddineLaouir/RIPOST>.

L'objectif n'est pas de vérifier la preuve de l'article — elle n'est pas en
cause — mais de mesurer **la perte de confidentialité effectivement exhibée
par l'implémentation**, et de la confronter au budget annoncé.

---

## Principe

RIPOST est traité comme une boîte noire `M(D, ε) → vue publiée`. On construit
deux jeux voisins `D` et `D' = D + {un enregistrement}`, on exécute le
mécanisme un grand nombre de fois sur chacun, puis on cherche un événement `S`
sur les sorties qui distingue les deux distributions.

La définition de la DP impose, pour tout `S` :

```
P[M(D) ∈ S]  ≤  exp(ε) · P[M(D') ∈ S] + δ
```

d'où une borne inférieure sur la perte réelle :

```
ε_emp = log( (p − δ) / q )
```

Deux précautions rendent ce nombre défendable plutôt qu'anecdotique :

1. **Bornes de Clopper–Pearson.** `p` est remplacé par une borne inférieure de
   confiance et `q` par une borne supérieure, de sorte que `ε_emp` reste
   valide malgré l'échantillonnage fini.
2. **Sélection du seuil sur des exécutions distinctes.** Le seuil de
   l'événement `S` est choisi sur une moitié des exécutions, et les bornes
   sont calculées sur l'autre moitié, jamais utilisée pour la sélection. Sans
   cette séparation, `ε_emp` mesurerait en partie le bruit du test lui-même.
   Une correction de Bonferroni est appliquée entre les statistiques.

---

## Statistiques auditées

Toutes sont observables sur la vue publiée.

| Statistique | Pourquoi elle est pertinente |
|---|---|
| `n_blocks` | La profondeur de décomposition dépend des données ; le nombre de blocs publiés en est la trace la plus directe. |
| `max_depth` | Même raison, exprimée en profondeur d'arbre. |
| `canary_mean` | Valeur bruitée du bloc contenant l'enregistrement qui distingue `D` de `D'`. |
| `canary_volume` | Taille du bloc l'absorbant : un bloc plus fin signale une décision de découpage différente. |
| `sum_estimate` | Signal agrégé sur toute la vue. |
| `max_block_mean` | Sensible aux valeurs extrêmes introduites par le bruit. |

`runtime_s` est mesuré mais **exclu par défaut** (`--include-timing` pour
l'activer) : le budget ε ne couvre que la sortie du mécanisme, pas ses
caractéristiques d'exécution. Confondre les deux dans un même chiffre
mélangerait deux canaux de fuite distincts.

---

## Placement de l'enregistrement témoin

Le résultat dépend fortement de l'endroit où l'enregistrement est ajouté,
puisque RIPOST décide de ses découpages à partir des données.

- `dense` — au cœur de la zone la plus peuplée. Cas témoin : l'enregistrement
  est noyé, la fuite devrait être minimale.
- `sparse` — dans une région vide et éloignée. Fait passer un sous-domaine de
  vide à non vide, distinction sur laquelle repose la première phase.
- `boundary` — dans une cellule vide immédiatement adjacente au cœur dense.
  Choix adversarial : il vise la condition de convergence, là où la présence
  d'un seul enregistrement peut faire basculer une décision de découpage.

La comparaison entre ces trois placements constitue en elle-même un résultat.

---

## Utilisation

```bash
git clone https://github.com/AlaEddineLaouir/RIPOST.git
pip install numpy pandas scipy

# 1. vérifier le wrapper — indispensable avant toute campagne
python mechanism.py --repo ./RIPOST --selftest

# 2. contrôle de validité du protocole : un écart de 50 enregistrements
#    doit être massivement détecté. Si ce n'est pas le cas, le harnais
#    est en cause et tout résultat négatif serait sans valeur.
python run_audit.py --repo ./RIPOST --runs 200 --n-canary 50 --epsilon 1.0

# 3. audit réel (un seul enregistrement d'écart)
python run_audit.py --repo ./RIPOST --runs 1000 --epsilon 0.1 0.5 1.0

# 4. volet canal auxiliaire, mesuré séparément
python run_audit.py --repo ./RIPOST --runs 500 --epsilon 1.0 --include-timing
```

Résultats écrits dans `results/audit_results.csv`, configuration dans
`results/audit_config.json`.

---

## Lecture des résultats

- **`ε_emp > ε_annoncé`** — violation au niveau de confiance retenu. À
  reproduire et documenter : paire de jeux, graine, statistique, seuil.
- **`ε_emp ≤ ε_annoncé`** — **non concluant**. Cela signifie que *ces*
  attaques, à *ce* nombre d'exécutions, n'ont pas trouvé de violation. Ce
  n'est en aucun cas une preuve de correction de l'implémentation.

---

## Limites connues

- L'audit ne couvre que les statistiques listées ci-dessus. Une fuite portée
  par une autre fonctionnelle de la sortie passerait inaperçue.
- Les jeux de données sont synthétiques et de petite taille, afin que
  plusieurs milliers d'exécutions restent réalisables.
- La borne obtenue est une borne inférieure : elle ne dit rien de la distance
  au budget réel lorsque aucune violation n'est trouvée.
- `max_depth` est reconstruit à partir de l'étendue des blocs, faute d'un
  accès direct à la profondeur dans la structure retournée.

---

## Note sur l'environnement

Le dépôt vise NumPy < 1.24 : `np.float`, `np.int` et `np.bool` y existaient
encore comme alias, supprimés depuis. `mechanism.py` les restaure avant
l'import, ce qui est la modification minimale permettant d'exécuter le code
publié sans le toucher.
