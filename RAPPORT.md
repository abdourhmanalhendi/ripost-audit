# Audit empirique de confidentialité différentielle — RIPOST

Preuve de concept réalisée à la demande d'A. Imine (LORIA), auteur de l'article
*RIPOST: Two-Phase Private Decomposition for Multidimensional Data*
(Laouir & Imine, ESORICS 2025, arXiv:2502.10207), sur le code publié à
<https://github.com/AlaEddineLaouir/RIPOST> (commit `97fa347`, package
`RIPOST.zip` → `RIPOST 2/src`).

L'objet de cet audit n'est pas la preuve de l'article, qui n'est pas en cause,
mais **l'implémentation** : mesurer la perte de confidentialité effectivement
exhibée par le code publié, la confronter au budget ε annoncé, et signaler
séparément tout bug de reproductibilité rencontré en cours de route.

---

## Résultat en une phrase

Le harnais d'audit initial rapportait `eps_emp = 4.64` pour `eps_claimed = 0.1`
via la statistique `sum_estimate`, sur une différence d'un seul enregistrement.
**Ce résultat était un artefact du harnais, pas une fuite de RIPOST** : il
lisait la valeur *non bruitée* renvoyée par `RIPOST.run`, pas la vue publiée.
Corrigé, et ré-audité à grande échelle (600 exécutions par jeu, contre 20-30
dans une première passe — voir §7 sur la taille d'échantillon), le test à
blanc donne `eps_emp = 0` avec des bornes maintenant serrées (`q_upper`
≤ 0.04 au lieu de ≤ 1.0), et l'audit réel (un enregistrement d'écart, six
statistiques, deux ε, trois placements) ne trouve **aucune violation**.

Deux résultats positifs, distincts de ce négatif, ont émergé du volet 2 :

1. **Échantillonneur Laplace en virgule flottante (Mironov, CCS 2012)** :
   82,4 % des sorties de `M(0)` sont *strictement inatteignables* par `M(1)`
   à l'échelle de bruit réelle de RIPOST (ε=0.1), avec témoins reproductibles
   — et cette propriété **survit** à la division et à l'addition de
   `perturb()` sur la valeur de bloc réellement publiée (73,7 % à 8,9 %
   selon la taille du bloc). C'est une violation démontrée sur la sortie
   publiée, pas seulement sur l'échantillonneur isolé. Voir §6.3.
2. **Canal auxiliaire temporel** : le temps d'exécution est corrélé au
   nombre de blocs publiés avec `r = 0.90–0.94` (`p ≈ 0`, n=1200) sur les
   trois placements à ε=0.1 — une corrélation quasi totale, bien plus forte
   que ce qu'un premier échantillon (n=60) avait laissé voir. Voir §6.4.

Les bugs #4 et #5 (le tirage de coupe et la perturbation finale ignorent tous
deux le `prng` fourni) sont confirmés avec numéro de ligne exact et une
démonstration reproductible en §5.

---

## 1. Méthodologie

RIPOST est traité comme une boîte noire `M(D, ε) → vue publiée`. On construit
deux jeux voisins `D` et `D' = D + {enregistrements}` puis on exécute `M` un
grand nombre de fois sur chacun, sous des graines indépendantes, et on
compare les distributions empiriques des sorties.

Pour un événement `S` sur les sorties, la DP impose
`P[M(D)∈S] ≤ exp(ε)·P[M(D')∈S] + δ`, d'où la borne inférieure
`eps_emp = log((p−δ)/q)`. Deux précautions rendent ce chiffre défendable :

1. **Bornes de Clopper–Pearson** : `p` est remplacé par sa borne inférieure de
   confiance, `q` par sa borne supérieure, à un niveau de confiance corrigé
   par Bonferroni entre les statistiques testées.
2. **Sélection sur échantillons disjoints** : le seuil de l'événement `S` (et,
   pour l'attaque apprise, la frontière de décision d'un classifieur) est
   choisi sur une partie des exécutions ; les bornes de confiance sont
   calculées sur une autre partie, jamais vue pendant la sélection.

Code : `mechanism.py` (wrapper boîte noire), `neighbors.py` (paires de jeux
voisins), `estimator.py` (bornes CP + recherche de seuil honnête),
`run_audit.py` (campagnes), `learned_attack.py` (attaque apprise),
`inspect_block.py` (diagnostic des attributs bruités), `float_attack.py`
(attaque de Mironov), `repro_demo.py` (démonstration des bugs #4/#5),
`timing_analysis.py` (canal auxiliaire).

**Environnement** : la première passe (voir historique git de ce dépôt)
tournait sous Windows, où `fork` n'est pas disponible — chaque exécution de
RIPOST y coûtait 4 à 8 s, ce qui limitait les campagnes à quelques dizaines
d'exécutions. Pour cet audit, une distribution WSL2 (Ubuntu, noyau Linux
6.18) a été installée sur la même machine spécifiquement pour disposer de
`fork` et de campagnes à grande échelle : Python 3.14.4, NumPy 2.5.3, pandas
3.0.5, scikit-learn 1.9.1, dans un environnement virtuel dédié
(`.venv-linux`). Débit mesuré : ~1,8 s par exécution de bout en bout
(processus + IPC inclus), le temps de calcul propre à RIPOST étant lui-même
de l'ordre de 0,3-0,6 s par appel à ε=0.1 (voir §6.4).

**Contrainte matérielle rencontrée** : la machine (16 Go de RAM, fortement
sollicitée par d'autres applications) n'a laissé qu'1,5 à 3 Go de mémoire
libre selon le moment. `RIPOST.run` crée un `multiprocessing.Manager()` et un
`Pool` neufs à *chaque* appel sans jamais les fermer explicitement — sur des
campagnes de plusieurs centaines d'appels, s'appuyer sur le ramasse-miettes
pour les récupérer s'est révélé trop lent : deux campagnes lancées à 4
exécutions concurrentes ont été tuées par le système pour manque de mémoire.
Correctif appliqué dans `mechanism.py` (pas dans le code de RIPOST) : forcer
`gc.collect()` après chaque appel. Le reste de l'audit a tourné avec 1 à 3
campagnes concurrentes selon la mémoire disponible à l'instant, jamais plus.
Cet aléa a coûté du temps mais n'affecte aucun résultat : chaque campagne
interrompue a simplement été relancée depuis zéro, sur les mêmes graines.

---

## 2. Le test à blanc, et l'artefact qu'il a débusqué

### 2.1 Ce que le harnais rapportait avant correction

Avant toute campagne, la consigne était de comparer `D` à lui-même : un audit
correct **doit** rapporter `eps_emp ≈ 0`, sans quoi il mesure son propre biais.
Le harnais initial ne passait pas ce test — c'est ainsi que le chiffre
`eps_emp = 4.64` (ε annoncé = 0.1, statistique `sum_estimate`, un seul
enregistrement d'écart, séparation quasi parfaite p=0.99/q=0.01) a été
identifié comme suspect : une séparation aussi nette, sur une différence
d'un enregistrement à ε=0.1, est incompatible avec un mécanisme de Laplace
correctement implémenté.

### 2.2 Cause racine, localisée dans le code publié

`RIPOST.run(...)` renvoie un couple `(p_view, block_result_list)` — les noms
sont ceux du driver des auteurs eux-mêmes (`src/RIPOST/ripost_run.py:105`),
qui n'utilise que `p_view` pour répondre aux requêtes. Les deux objets ne
sont **pas interchangeables** :

- `block_result_list` (`out[1]`) contient des `BlockResult` dont le champ
  `.mean` est la moyenne empirique **brute**, calculée par
  `calculate_ae_from_data`/`calculate_mean_and_aggregation_error`
  (`count_table.py:606-633`) directement sur les données, avant tout bruit.
- Le bruit de Laplace *est* tiré — `pe = np.random.laplace(0,
  sensitivity/epsilon_p)` (`RIPOST.py:90`) — et stocké dans un champ séparé,
  `block_result.perturbation_error`, mais **jamais additionné** à
  `block_result.mean` sur cet objet.
- Le bruit n'est appliqué que dans `NoisedCountTable.__init__`
  (`count_table.py:441-447`), qui construit `p_view.blocks` comme
  `Block(domaine, mean + perturbation_error/block_size)` via la méthode
  `perturb()` (`count_table.py:525-527`). `p_view` (`out[0]`) est donc la vue
  *publiée et bruitée* ; `out[1]` n'est qu'un échafaudage pré-bruit qui se
  trouve être retourné aussi.

La version initiale de `mechanism.py` lisait `out[1]`. Chaque statistique
calculée à partir de `block_means` (`sum_estimate`, `canary_mean`,
`max_block_mean`) lisait donc **la donnée brute**, pas la sortie publiée —
ce qui explique mécaniquement une séparation quasi parfaite entre `D` et
`D'` : ce n'était pas du bruit de Laplace insuffisant, c'était l'absence
totale de bruit sur la statistique observée.

### 2.3 Preuve directe : quels attributs sont réellement bruités

`inspect_block.py` exécute le mécanisme deux fois sur les mêmes données et
liste quels attributs varient. Après correction (lecture de `p_view.blocks`,
des objets `Block` qui n'exposent que `ranges`, `size`, `value`) :

```
epsilon=1.0   |D|=800
  size    varie  -> dépend des données/du bruit (structure de coupe)
  value   varie  -> bruité, utilisable
n_blocks: run1=32  run2=53  -> la structure dépend des données/du bruit
```

Reproductible par : `python inspect_block.py ./RIPOST/unzipped 1.0`.

### 2.4 Ampleur du bruit (plancher de bruit)

Cinq exécutions indépendantes sur le même `D` (`|D|=800`, ε=1.0,
`sum_estimate = Σ value·size`) : moyenne 805,955, écart-type 9,476, pour un
`|D|` vrai de 800. Un écart d'un enregistrement (±1) est donc noyé dans un
bruit d'échelle ~9,5 à ε=1.0. Reproductible par
`python inspect_block.py ./RIPOST/unzipped 1.0`.

### 2.5 Test à blanc après correction, à grande échelle

`python run_audit.py --repo ./RIPOST/unzipped --null-test --runs 600 --epsilon 0.1 --rows 800 --shape 6 6 3 --placements boundary --seed 0 --out results_v2/null_eps01`

D' := D (copie exacte), n=600 exécutions par jeu (300 en évaluation après
séparation), ε=0.1 :

| statistique | eps_emp | p_lower | q_upper | direction |
|---|---|---|---|---|
| n_blocks | 0.0000 | 0.0000 | 0.0226 | D'>D |
| max_depth | 0.0000 | 0.9242 | 0.9844 | D'>D |
| canary_mean | 0.0000 | 0.0026 | 0.0339 | D'>D |
| canary_volume | 0.0000 | 0.0014 | 0.0440 | D>D' |
| sum_estimate | 0.0000 | 0.0000 | 0.0158 | D'>D |
| max_block_mean | 0.0000 | 0.0000 | 0.0158 | D>D' |

Même chose à ε=1.0 (`--epsilon 1.0 --seed 1`, `results_v2/null_eps1`) :

| statistique | eps_emp | p_lower | q_upper | direction |
|---|---|---|---|---|
| n_blocks | 0.0000 | 0.0000 | 0.0158 | D'>D |
| canary_mean | 0.0000 | 0.0000 | 0.0158 | D'>D |
| canary_volume | 0.0000 | 0.0000 | 0.0226 | D>D' |
| sum_estimate | 0.0000 | 0.0041 | 0.0390 | D>D' |
| max_block_mean | 0.0000 | 0.0000 | 0.0158 | D'>D |

**`eps_emp = 0` partout, avec des bornes désormais serrées** (`q_upper` ne
dépasse plus 0,044, contre jusqu'à 1,0 dans la première passe à 20-30
exécutions). C'est la différence essentielle avec la version précédente de ce
rapport : un `q_upper` proche de 1 rend `eps_emp = 0` presque automatique,
qu'il y ait ou non une fuite réelle à détecter ; un `q_upper` de l'ordre de
0,02-0,04 signifie que le test avait une vraie chance de détecter une
séparation et n'en a pas trouvé.

---

## 3. Audit réel après correction (un enregistrement d'écart), à grande échelle

`python run_audit.py --repo ./RIPOST/unzipped --runs 600 --epsilon 0.1 --rows 800 --shape 6 6 3 --placements {boundary,sparse,dense} --seed 2 --dump-raw --out results_v2/real_eps01_<placement>`
(une invocation séparée par placement, même graine, pour l'exécution parallèle)

`eps_emp` par statistique et par placement, ε=0.1, n=600 (`q_upper` entre
parenthèses) :

| statistique | boundary | sparse | dense |
|---|---|---|---|
| n_blocks | 0.0000 (0.016) | 0.0000 (0.023) | 0.0000 (0.023) |
| max_depth | 0.0000 (1.000) | 0.0000 (0.989) | 0.0000 (0.997) |
| canary_mean | 0.0000 (0.089) | 0.0000 (0.016) | 0.0000 (0.101) |
| canary_volume | 0.0000 (0.023) | 0.0000 (0.072) | 0.0000 (0.058) |
| sum_estimate | 0.0000 (0.044) | 0.0000 (0.039) | 0.0000 (0.016) |
| max_block_mean | 0.0000 (0.016) | 0.0000 (0.063) | 0.0000 (0.023) |

Et à ε=1.0, placement `boundary` uniquement (`--epsilon 1.0 --seed 2`,
`results_v2/real_eps1_boundary`) :

| statistique | eps_emp | q_upper |
|---|---|---|
| n_blocks | 0.0000 | 0.029 |
| canary_mean | 0.0000 | 0.084 |
| canary_volume | 0.0000 | 0.016 |
| sum_estimate | 0.0000 | 0.016 |
| max_block_mean | 0.0000 | 0.093 |

**Aucune violation trouvée, sur aucune statistique, sur aucun placement, à
aucun des deux ε**, y compris `sum_estimate` à ε=0.1 — l'attaque et le budget
exacts qui avaient initialement produit `eps_emp = 4.64`. Avec des bornes
`q_upper` maintenant toutes sous 0,10 (`max_depth` excepté, structurellement
proche de 1 partout — voir limites), ce résultat négatif porte davantage de
poids que dans la première passe.

---

## 4. Contrôle de puissance et recherche du plus petit k détectable

Le test à blanc et l'audit réel montrent tous deux `eps_emp=0`. Sans preuve
que le harnais *peut* détecter une vraie différence, ce zéro ne prouve rien.
`n_canary=k` construit `D'=D+k` enregistrements identiques au même endroit ;
ce n'est **pas** une revendication de confidentialité (l'unité de la DP est 1
enregistrement) mais un contrôle du harnais lui-même — d'où la comparaison à
`k·ε`, pas à `ε`.

`python run_audit.py --repo ./RIPOST/unzipped --runs 600 --n-canary k --epsilon 1.0 --rows 800 --shape 6 6 3 --placements boundary --seed <3..9> --out results_v2/power_k<k>`

| k | eps_emp max | statistique | k·ε applicable | verdict |
|---|---|---|---|---|
| 1 | 0.0000 | — | 1.0 | non détecté (§3, à ε=1.0 aussi) |
| 2 | 0.0000 | — | 2.0 | **non détecté** |
| 3 | 0.4807 | canary_mean | 3.0 | **détecté** |
| 5 | 1.7694 | canary_mean | 5.0 | détecté |
| 15 | 3.1097 | canary_mean | 15.0 | détecté |
| 50 | 4.1007 | canary_mean | 50.0 | détecté |

Le plus petit écart détecté par ce harnais, sur ce jeu de données et ce
placement, est **k=3** : `canary_mean` franchit le seuil de détection
(`eps_emp=0.48 > 0.05`) à k=3 mais pas à k=2 (`eps_emp=0.0000`,
`q_upper=0.044`) ni à k=1. `sum_estimate` suit la même tendance mais plus
lentement (0 à k=3, 0.24 à k=5, 0.77 à k=15, 3.25 à k=50) : la sensibilité
dépend de la statistique choisie, `canary_mean` est ici la plus puissante.
Ceci resserre franchement l'écart laissé ouvert dans la version précédente de
ce rapport (« entre 1 et 15, non recherché par dichotomie ») : la frontière
est maintenant localisée entre k=2 et k=3, sur ce jeu de paramètres précis.

---

## 5. Bugs de reproductibilité (code RIPOST publié)

Ces éléments sont des défauts du code publié, indépendants de toute question
de confidentialité — listés séparément comme demandé. Les bugs #4 et #5 ont
été relus ligne par ligne pour cette révision et sont confirmés exacts.

1. **`CountTable.from_dataset` laisse `mesure_table=None`**
   (`count_table.py:360`) : tout appel ultérieur à `RIPOST.run` échoue
   (`'NoneType' object is not subscriptable`) sans contournement manuel (déjà
   nécessaire dans le driver des auteurs eux-mêmes,
   `src/RIPOST/ripost_run.py`, et reproduit dans `mechanism.py`).
2. **`CountTable.from_pd_table` / `from_pd_count_table`**
   (`count_table.py:363-402`) appellent `CountTable(...)` avec 4 arguments
   positionnels alors que `__init__` en exige 5 : `TypeError` systématique,
   ces deux constructeurs sont inutilisables tels quels.
3. **`RIPOST.run`, tirage de la graine du bloc** (`RIPOST.py:24`) :
   `prng.randint(0, 2949672950)` sans `dtype` explicite. Le type par défaut de
   NumPy pour `randint` est le `long` C de la plateforme — 64 bits sous
   Linux/macOS, 32 bits sous Windows — et la constante dépasse `int32` :
   `ValueError: high is out of bounds for int32` sur toute machine Windows,
   succès silencieux ailleurs. Contourné ici par une sous-classe
   `WideRandomState` qui force `dtype=np.int64` (reproduit exactement ce que
   fait déjà, sans y penser, toute plateforme à `long` 64 bits).
4. **`exp_mech` ignore le `prng` reçu — confirmé, ligne exacte.**
   `src/RIPOST/RIPOST.py:202-221`, fonction `exp_mech(prng, eps, scores,
   targets, sensitivity=1)` :
   - ligne 207 (`sensitivity == 0`) : `index = prng.choice(len(targets))` —
     correct, mais **cette branche n'est jamais atteinte en pratique**. Le
     seul site d'appel (`cut_exp_mech_population`, ligne 268) initialise
     `sensitivity=1` et ne le remet à 0 que si `block.size()==1`
     (`sensitivity = 2*(2-2/block.size())`) — or un bloc de taille 1 a
     `cardinality==1` sur toutes ses dimensions, donc la boucle qui remplit
     `scores`/`targets` (`if cardinality > 1:`) ne s'exécute jamais, et
     `exp_mech` n'est même pas appelé (retour anticipé `return 0`,
     `RIPOST.py:278`). Vérifié en lisant `cut_exp_mech_population` en entier
     (`RIPOST.py:225-278`).
   - ligne 216 (`sensitivity != 0`, la branche réellement empruntée à
     **chaque** appel) : `rand_i = np.random.choice(possibs, 1,
     p=cum_weights)[0]` — le générateur **global** de NumPy, pas `prng`.
   - **Démonstration reproductible** (`repro_demo.py`) : `prng` fixé à la
     graine 42 sur les deux appels, seule la graine globale change
     (`np.random.seed(111)` puis `np.random.seed(222)`) :
     ```
     np.random.seed=111  prng_seed=42(fixe)  ->  n_blocks=38
     np.random.seed=222  prng_seed=42(fixe)  ->  n_blocks=16
     ```
     Sorties complètement différentes (38 vs 16 blocs, moyennes de bloc
     disjointes) malgré un `prng` identique : le tirage de coupe du
     mécanisme exponentiel — la décision la plus sensible de tout
     l'algorithme — n'est pas reproductible depuis la graine que l'appelant
     contrôle.
5. **`RIPOST.run`, perturbation finale des feuilles — confirmé, ligne
   exacte.** `RIPOST.py:90`, à l'intérieur de `run()` (définie ligne 10,
   `prng` est un paramètre de cette fonction et reste en portée) :
   `pe = np.random.laplace(0, sensitivity / epsilon_p)` — le générateur
   global, alors que `prng` est disponible et n'est utilisée nulle part dans
   cette boucle de finalisation. Le même `repro_demo.py` capture cet effet
   conjointement avec le bug #4 (les deux se produisent dans le même appel à
   `run()` et ne sont pas isolables sans modifier le code publié).
6. **Parallélisme mort-né** : `MAX_PROCESS = multiprocessing.cpu_count()-1`
   est calculé puis jamais utilisé ; le pool réellement créé est
   `multiprocessing.Pool(1)`. Malgré le `Manager`/`Queue` mis en place, aucune
   parallélisation n'a jamais lieu.
7. **`exp_mech`** attrape tout (`except Exception: print(...); return 0`) :
   une erreur numérique (dépassement de capacité dans `np.exp`, par exemple)
   est silencieusement interprétée comme « pas de coupe », sans remontée
   d'erreur.
8. **Fuite de ressources par appel** : `RIPOST.run` crée un
   `multiprocessing.Manager()` neuf à chaque appel sans jamais appeler
   `manager.shutdown()` — voir §1, « Contrainte matérielle rencontrée ». Sur
   une campagne de centaines d'exécutions dans le même processus, ces objets
   s'accumulent plus vite que le ramasse-miettes ne les récupère.

Bugs propres au harnais d'audit, corrigés en cours d'étude et documentés ici
par transparence :

9. **`mechanism.py` lisait `out[1]` au lieu de `out[0]`** — voir §2.2. C'est
   la cause unique du chiffre `eps_emp=4.64` initialement rapporté.
10. **`inspect_block.py` n'avait pas de garde `if __name__=="__main__"`** :
    sous Windows (`spawn`), chaque appel à `RIPOST.run` (qui crée un
    `multiprocessing.Manager`) réimportait et ré-exécutait le script entier
    dans le sous-processus, ce que la vérification `_check_not_importing_main`
    de Python refuse — provoquant un blocage de plusieurs minutes avant
    l'échec. Corrigé en encapsulant tout dans `main()`.
11. **`mechanism.py` ne fermait pas les ressources de RIPOST** (bug #8
    ci-dessus, côté audit) — corrigé par un `gc.collect()` explicite après
    chaque appel.

---

## 6. Volet 2 — élargissement de la surface d'attaque

### 6.1 Traçage du budget ε (analytique)

Avec les hyperparamètres des auteurs (`ratio`=0.3, `gamma`=0.9,
`phase_split`=0.4, `step`=1, appel `run(table, eps, .3, prng, 0, 0, .9, [.4,
1])`) :

```
epsilon_r     = 0.3 · eps
epsilon_p     = 0.7 · eps                      (perturbation finale)
epsilon_cut   = 0.03 · eps                     (sélection de coupe)
epsilon_conv  = 0.27 · eps                     (test de convergence)
epsilon_p + epsilon_cut + epsilon_conv = eps   exactement
```

Le budget de coupe/convergence est ensuite réparti sur la profondeur
inconnue a priori de la décomposition via la série
`epsilon_cut_per_depth = budget · 4/(d(d+1))`, `d = profondeur+3`
(`RIPOST.py`, `cut_exp_mech_population`, `random_converge_pop_lap`,
`random_converge_lap`). Cette série télescope :
`Σ_{d=4}^{∞} 4/(d(d+1)) = 1` exactement, donc le budget réellement consommé
sur une profondeur *finie* quelconque reste strictement inférieur au budget
alloué, quelle que soit la profondeur atteinte — cohérent avec l'objectif
annoncé dans l'article (gestion du budget sans connaître `h` à l'avance,
résumé de l'abstract arXiv:2502.10207). **Limite inchangée** : le texte/les
équations de la section 5.2 elle-même n'ont pas pu être extraits (PDF non
exploitable par l'outil de récupération disponible) ; cette vérification est
faite au niveau du code et de l'objectif énoncé dans le résumé, pas par
confrontation littérale avec la preuve du papier.

Les tirages de bruit du test de convergence (`block.prng.laplace(...)`)
utilisent correctement le `prng` du bloc, dérivé de la graine contrôlée — au
contraire du tirage de coupe et de la perturbation finale (bugs #4 et #5,
§5).

### 6.2 Isolation des sous-mécanismes, à grande échelle

Isoler un sous-mécanisme en appelant directement `exp_mech` ou
`random_converge_*` exigerait de reconstruire à la main leur état interne —
au risque de mal réimplémenter la mécanique et d'auditer notre propre
reconstruction plutôt que le code publié. On isole donc approximativement via
l'hyperparamètre public `ratio` de `RIPOST.run` (`--ratio` ajouté à
`run_audit.py`), qui déplace la masse du budget d'un côté ou de l'autre sans
toucher au code : voir limite en §7.

`python run_audit.py --repo ./RIPOST/unzipped --ratio {0.02,0.98} --epsilon 0.1 --runs 600 --placements boundary --seed {7,8}`

**`ratio=0.02`** (budget de décomposition minimal, `epsilon_p ≈ 0.98·eps` —
la perturbation finale des feuilles reçoit presque tout le budget, isolant le
mécanisme de Laplace) :

| statistique | eps_emp | q_upper |
|---|---|---|
| n_blocks | 0.0000 | 0.044 |
| max_depth | 0.0000 | 0.867 |
| canary_mean | 0.0000 | 0.016 |
| canary_volume | 0.0000 | 0.049 |
| sum_estimate | 0.0000 | 0.016 |
| max_block_mean | 0.0000 | 0.023 |

**`ratio=0.98`** (budget de décomposition maximal, perturbation finale quasi
annulée — isole le test de convergence et la sélection de coupe) :

| statistique | eps_emp | q_upper |
|---|---|---|
| n_blocks | 0.0000 | 0.029 |
| max_depth | 0.0000 | 1.000 |
| canary_mean | 0.0000 | 0.067 |
| canary_volume | 0.0000 | 0.023 |
| sum_estimate | 0.0000 | 0.016 |
| max_block_mean | 0.0000 | 0.016 |

**Aucune violation dans les deux configurations extrêmes**, à n=600 et avec
des bornes désormais serrées. Ni le mécanisme de Laplace des feuilles pris
quasiment seul, ni le test de convergence/la sélection de coupe pris
quasiment seuls, ne dépassent leur budget alloué sur ces campagnes.

### 6.3 Analyse en virgule flottante — violation démontrée (Mironov, CCS 2012)

RIPOST tire tous ses bruits via `np.random.laplace` / `prng.laplace`
(générateur historique `RandomState` de NumPy), sans aucune atténuation
(« snapping mechanism », arithmétique exacte, ou dithering). C'est exactement
la construction que Mironov analyse comme vulnérable, et non un défaut propre
à RIPOST : ce qui suit établit *par construction* que ce code précis est dans
cette classe, quantifie l'effet à l'échelle de bruit réelle de RIPOST, et
fournit des témoins reproductibles.

**Méthode** (`float_attack.py`) : `NumPy`'s `RandomState.laplace` est
réimplémenté exactement — `U = k/2^53` (k entier), `v = scale·log(U+U)` si
`U<0.5`, sinon `v = -scale·log(2-2U)`. `is_reachable(v)` inverse cette
formule pour retrouver l'indice `k` requis, cherche dans une fenêtre de `k`
voisins, et vérifie l'égalité *exacte* en flottant. **Auto-test avant toute
conclusion** : les 20 000/20 000 tirages réels d'un `RandomState` authentique
sont reconnus comme atteignables — condition nécessaire avant de faire
confiance au reste.

**1. Support borné.** Pour l'échelle de bruit réelle de RIPOST à ε=0.1
(`epsilon_p = 0.7×0.1 = 0.07`, échelle `1/epsilon_p ≈ 14.29`) :

```
plafond théorique  scale · ln(2^53)  = 524.81
plus grand tirage observé (100 000 tirages) = 170.36
```

Un vrai Laplace a un support infini ; celui-ci a un plafond dur. Toute
observation au-delà est de probabilité **exactement nulle** dans
l'implémentation — ce qui à lui seul contredit la définition d'ε-DP pur
(support plein requis).

**2. Témoins sur l'échantillonneur brut** : sorties de `M(0) = 0 + Lap(scale)`
inatteignables par `M(1) = 1 + Lap(scale)`, 100 000 tirages, `scale=14.29`,
graine 0 :

```
inatteignables : 82 409 / 100 000  (82.41 %)
premiers témoins (bits exacts) :
  1.467533251473782    0x1.77b04258df581p+0
  3.286799503487481    0x1.a4b5d89bfe6f3p+1
  1.34362816796179     0x1.57f803ff615a4p+0
```
Observer l'un de ces témoins identifie l'entrée avec certitude : la perte de
confidentialité y est non bornée, quel que soit l'ε annoncé.

**3. Le témoin survit-il à la valeur de bloc réellement publiée ?**
`NoisedCountTable.__init__` construit chaque feuille comme
`Block(domaine, perturb(domaine, perturbation_error, mean))`, et `perturb()`
(`count_table.py:525-527`) calcule `mean + (noise / block_size)` — une
division et une addition de plus, qui rearrondissent. `mean1 = mean0 +
1/block_size` est la sensibilité exacte de RIPOST lui-même pour un
enregistrement ajouté (la somme des comptes de cellules du bloc augmente
d'exactement 1 ; `block_size` est le même diviseur que dans `perturb()`).

Le modèle composé (mean, bruit, division) a d'abord échoué son propre
auto-test (un algorithme d'inversion trop naïf ratait le bon indice `k` dans
~2 à 10 % des cas) — corrigé en cherchant directement sur les indices `k`
plutôt que sur les doubles voisins de la valeur candidate ; l'auto-test passe
ensuite à 2000/2000 pour `block_size` ∈ {1, 27}. Résultat, `mean0=5.0`,
100 000 tirages, ε=0.1 :

| block_size | mean1 | témoins survivants |
|---|---|---|
| 1 | 6 | 73 690 / 100 000 (73.69 %) |
| 4 | 5.25 | 48 371 / 100 000 (48.37 %) |
| 27 | 5.037… | 8 917 / 100 000 (8.92 %) |

La fraction diminue avec la taille du bloc (plus de division = plus
d'arrondi = moins de témoins qui survivent exactement), mais reste non nulle
et substantielle dans les trois cas. **C'est donc une violation démontrée sur
la quantité que RIPOST publie réellement** (`p_view.blocks[i].value`), pas
seulement sur l'échantillonneur Laplace pris isolément.

**Portée, honnêtement.** La classe de vulnérabilité est celle de Mironov
(CCS 2012) et touche toute implémentation naïve du mécanisme de Laplace en
virgule flottante sans atténuation — RIPOST n'est pas visé spécifiquement, il
n'est simplement pas exempté. Les campagnes statistiques à échantillon fini
des sections précédentes (quelques centaines d'exécutions) ne peuvent ni ne
cherchent à démontrer cet effet : il a fallu construire les témoins par
inversion exacte de l'échantillonneur, pas par un test d'hypothèse sur des
tirages aléatoires. Reproductible par : `python float_attack.py --epsilon 0.1 --samples 100000`.

### 6.4 Canal auxiliaire — temps d'exécution, résultat promu

`runtime_s` est mesuré mais exclu de l'estimation d'ε par défaut (le budget
de confidentialité couvre la sortie, pas le profil d'exécution). Ré-analysé
sur les campagnes à 600 exécutions par jeu du §3 (`--dump-raw`), donc n=1200
par ligne (D et D' confondus) au lieu de n=60 dans la version précédente :

`python timing_analysis.py 'results_v2/*/raw_*.csv'`

| config | n | corr(runtime, n_blocks) | corr(runtime, max_depth) | Mann-Whitney D vs D' (p) |
|---|---|---|---|---|
| ε=0.1, boundary | 1200 | **+0.914** (p≈0) | +0.258 (p=9×10⁻²⁰) | 0.069 |
| ε=0.1, dense | 1200 | **+0.938** (p≈0) | +0.277 (p=1×10⁻²²) | 0.120 |
| ε=0.1, sparse | 1200 | **+0.903** (p≈0) | +0.262 (p=2×10⁻²⁰) | 0.897 |
| ε=1.0, boundary | 1200 | +0.568 (p=1×10⁻¹⁰³) | n/a (constant) | 0.539 |

**Le temps d'exécution est corrélé au nombre de blocs publiés avec une force
quasi totale** (r=0.90 à 0.94, p pratiquement nul) sur les trois placements à
ε=0.1 — nettement plus net que le r=0.63 mesuré sur le premier échantillon
(n=60). Le nombre de blocs est une structure dépendante des données ; le
temps d'exécution la révèle sans aucune protection différentielle puisqu'il
n'est pas bruité. C'est un canal auxiliaire réel, indépendant du budget ε.

**Sur la question spécifique d'un enregistrement d'écart** : le test de
Mann-Whitney comparant directement le temps d'exécution entre `D` et `D'`
reste **non significatif au seuil 0,05 sur les quatre configurations**, même
à n=1200 (contre n=60 dans la version précédente). Le cas le plus proche du
seuil est `boundary` à ε=0.1 (`p=0.069`) — plus proche qu'avec le petit
échantillon, mais toujours pas concluant. **Réponse explicite à la question
posée** : passer de n=60 à n=1200 n'a *pas* rendu ce test significatif ; il
l'a rapproché du seuil sur le placement le plus favorable (`boundary`), ce
qui est un indice, pas une preuve, qu'un échantillon encore plus grand ou un
test mieux ciblé (par exemple restreint aux blocs contenant la cellule
témoin, ou à un ε plus petit où la profondeur varie davantage) pourrait
franchir le seuil. C'est la piste la plus prometteuse laissée ouverte par cet
audit — voir conclusion.

### 6.5 Attaque apprise (DP-Sniper-lite), à grande échelle

Implémentation dans `learned_attack.py` : vecteur de traits = les six
statistiques choisies à la main + moyennes et volumes de blocs triés et
tronqués/complétés à une longueur fixe (16), régression logistique
(scikit-learn) après standardisation. Trois découpes disjointes des
exécutions : entraînement du classifieur (50 %), puis sélection du seuil et
évaluation finale sur les 50 % restants via `estimator.audit_statistic`.

`python learned_attack.py --repo ./RIPOST/unzipped --epsilon 0.1 --runs 600 --rows 800 --shape 6 6 3 --placement boundary --seed 9` :

```
p_hat=0.0333   q_hat=0.06   p_lower=0.0132   q_upper=0.1024
n_D=150   n_Dprime=150   (contre n=20 dans la version précédente)
eps_emp = 0.0000   ->  aucune violation
```

Aucune violation trouvée non plus par cette attaque apprise, à `ε=0.1`, avec
un échantillon d'évaluation near 8× plus grand que la première passe — bornes
plus serrées, conclusion inchangée.

---

## 7. Limites

- **Statistiques couvertes** : `n_blocks`, `max_depth`, `canary_mean`,
  `canary_volume`, `sum_estimate`, `max_block_mean`, plus une statistique
  apprise (régression logistique sur un vecteur de traits, §6.5). Une fuite
  portée par une autre fonctionnelle de la sortie ne serait pas vue.
- **`max_depth` reste peu informatif** : son `q_upper` avoisine 1,0 dans
  presque toutes les campagnes (la profondeur maximale atteinte est quasiment
  toujours la même valeur), ce qui n'est pas un défaut du harnais mais une
  propriété du jeu de données/domaine choisi (6×6×3, 800 lignes) — la
  profondeur sature avant que le budget ne s'épuise. Un domaine plus grand
  exercerait mieux cette statistique.
- **Grille testée, et grille non testée par manque de temps.** Chaque config
  a tourné à `--runs 600` comme demandé, sur `rows=800`, `shape=(6,6,3)`. Le
  contrôle de puissance et l'isolation n'ont été menés que sur le placement
  `boundary` (le plus adversarial). La bissection sur k s'est arrêtée à une
  résolution de 1 (k=2 non détecté, k=3 détecté) : suffisant pour répondre à
  la question posée, mais propre à ce jeu de paramètres précis — un autre
  `rows`/`shape`/seed donnerait probablement une frontière différente.
- **Contrainte mémoire (§1)** a limité la concurrence à 1-3 campagnes
  simultanées selon le moment, jamais plus ; deux campagnes à 4 exécutions
  concurrentes ont été tuées par le système et relancées. Aucun résultat n'a
  été calculé à partir d'une exécution partielle : chaque table de ce
  rapport provient d'une campagne qui s'est terminée normalement et a écrit
  son `audit_results.csv`.
- **Un résultat négatif ne prouve rien.** `eps_emp ≤ eps_claimed` signifie que
  *ces* attaques, à *cette* taille d'échantillon (désormais bien plus grande
  qu'initialement, mais toujours finie), n'ont pas trouvé de violation —
  jamais que l'implémentation est correcte.
- **Budget ε (§6.1)** vérifié au niveau du code et de l'objectif énoncé dans
  le résumé de l'article, pas par relecture littérale de la preuve de la
  section 5.2 (PDF non exploitable par l'outil disponible dans cette
  session).
- **Mironov (§6.3)** : la survie du témoin à la valeur publiée est démontrée
  *par construction* (inversion exacte de l'échantillonneur composé avec
  `perturb()`), pour des valeurs de `mean`/`block_size` représentatives mais
  choisies à la main (`mean0=5.0`, `block_size` ∈ {1,4,27}) — pas balayées
  exhaustivement sur toute la plage réellement observée en campagne. C'est
  néanmoins une démonstration constructive, pas une extrapolation
  statistique : chaque témoin cité est vérifié bit à bit.
- **Isolation des sous-mécanismes (§6.2)**, via des valeurs extrêmes de
  l'hyperparamètre `ratio` plutôt que par appel direct aux fonctions internes
  — un choix qui évite de mal réimplémenter la mécanique interne, mais qui
  n'isole qu'approximativement chaque sous-mécanisme (`ratio`→0 et `ratio`→1
  partagent le même partitionnement coupe/convergence).
- **Canal temporel (§6.4)** : la corrélation runtime~n_blocks est
  extrêmement solide : le test D-vs-D' directement sur le timing, lui, reste
  non concluant à ε=0.1/boundary (p=0.069) — état des lieux honnête, ni
  confirmé ni infirmé à ce stade.

---

## 8. Conclusion

Le résultat central de cet audit est négatif et robuste : après correction
d'un bug du harnais qui lisait la sortie non bruitée de RIPOST, aucune des
campagnes menées — test à blanc, audit à un enregistrement sur trois
placements et deux ε, isolation par sous-mécanisme, attaque par classifieur —
ne trouve de violation du budget ε annoncé, désormais avec des bornes de
confiance suffisamment serrées (`q_upper` typiquement sous 0,05, contre
jusqu'à 1,0 dans la première passe) pour que ce négatif ait un sens
statistique réel. Le contrôle de puissance situe la sensibilité du harnais à
k=3 enregistrements sur ce jeu de paramètres. Deux résultats positifs,
distincts de l'audit statistique, ont été établis par construction plutôt que
par test d'hypothèse : l'échantillonneur Laplace de RIPOST est démontrablement
dans la classe de vulnérabilité de Mironov (CCS 2012), avec des témoins qui
survivent jusqu'à la valeur de bloc publiée ; et deux bugs de reproductibilité
significatifs (le tirage de coupe et la perturbation finale ignorent tous
deux le générateur aléatoire que l'appelant contrôle) sont confirmés avec
numéro de ligne exact et démonstration reproductible. Ce qui n'a pas été
établi : que le canal temporel ou l'attaque de Mironov constituent une
violation *mesurable* du budget ε annoncé par une campagne statistique à
échantillon fini — les deux restent des risques structurels confirmés, pas
des violations chiffrées de l'ε revendiqué. La piste la plus prometteuse pour
la suite est le canal temporel : la corrélation runtime~structure est quasi
totale (r>0.9) et le test direct D-vs-D' sur le timing, à p=0.069 sur le
placement le plus favorable avec n=1200, est proche du seuil de significativité
sans le franchir — un échantillon plus grand, ou un test restreint aux blocs
contenant la cellule témoin, mériterait d'être tenté avant de conclure.

---

## 9. Reproduire

```bash
git clone https://github.com/AlaEddineLaouir/RIPOST.git
cd RIPOST && unzip RIPOST.zip -d unzipped   # -> unzipped/RIPOST 2/src

# 0. valider le wrapper
python mechanism.py --repo ./RIPOST/unzipped --selftest

# 1. test à blanc, n=600 (doit rapporter eps_emp=0 partout, q_upper serré)
python run_audit.py --repo ./RIPOST/unzipped --null-test --runs 600 \
    --epsilon 0.1 --rows 800 --shape 6 6 3 --placements boundary --seed 0 \
    --out results_v2/null_eps01
python run_audit.py --repo ./RIPOST/unzipped --null-test --runs 600 \
    --epsilon 1.0 --rows 800 --shape 6 6 3 --placements boundary --seed 1 \
    --out results_v2/null_eps1

# 2. audit réel, n=600, par placement (une invocation par placement)
for p in boundary sparse dense; do
  python run_audit.py --repo ./RIPOST/unzipped --runs 600 --epsilon 0.1 \
      --rows 800 --shape 6 6 3 --placements $p --seed 2 --dump-raw \
      --out results_v2/real_eps01_$p
done
python run_audit.py --repo ./RIPOST/unzipped --runs 600 --epsilon 1.0 \
    --rows 800 --shape 6 6 3 --placements boundary --seed 2 --dump-raw \
    --out results_v2/real_eps1_boundary

# 3. contrôle de puissance et bissection
for k in 2 3 5 15 50; do
  python run_audit.py --repo ./RIPOST/unzipped --runs 600 --n-canary $k \
      --epsilon 1.0 --rows 800 --shape 6 6 3 --placements boundary \
      --seed $((k+2)) --out results_v2/power_k$k
done

# 4. attaque apprise
python learned_attack.py --repo ./RIPOST/unzipped --epsilon 0.1 --runs 600 \
    --rows 800 --shape 6 6 3 --placement boundary --seed 9

# 5. isolation approximative des sous-mécanismes
python run_audit.py --repo ./RIPOST/unzipped --ratio 0.02 --epsilon 0.1 \
    --runs 600 --placements boundary --seed 7 --out results_v2/isolate_leaf
python run_audit.py --repo ./RIPOST/unzipped --ratio 0.98 --epsilon 0.1 \
    --runs 600 --placements boundary --seed 8 --out results_v2/isolate_ccss

# 6. attaque de Mironov (échantillonneur + valeur publiée)
python float_attack.py --epsilon 0.1 --samples 100000

# 7. démonstration des bugs #4/#5 (non-reproductibilité depuis la graine)
python repro_demo.py

# 8. canal temporel
python timing_analysis.py 'results_v2/*/raw_*.csv'
```

Toutes les graines sont fixées ; `WideRandomState` (mechanism.py) et le
`gc.collect()` explicite après chaque appel sont les deux seuls
contournements nécessaires pour exécuter le code publié tel quel à grande
échelle. Les CSV bruts de chaque campagne sont dans `results_v2/`.
