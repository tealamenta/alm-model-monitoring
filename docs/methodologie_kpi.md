# Méthodologie : définition et justification des KPIs de monitoring

Ce document répond au premier axe de la mission (« cadrage et définition de la stratégie
de monitoring ») : pour chaque indicateur, **ce qu'il mesure**, **pourquoi il est pertinent
pour un modèle ALM**, **comment ses seuils sont fixés** et **ce qu'il ne détecte pas**.
Les valeurs chiffrées renvoient au cas d'application (modèle de remboursement anticipé
Lending Club, `configs/lending_club_prepayment.yaml`).

## 1. Cadre général

### 1.1 Ce qu'on surveille, et dans quel ordre

Un modèle comportemental ALM (remboursement anticipé, écoulement des dépôts à vue,
propension au dépôt à terme…) peut se dégrader de trois façons, qui ne se détectent pas
au même moment :

| Famille | Question | Disponibilité | Rôle |
|---|---|---|---|
| **Qualité des données en entrée** | Les données reçues sont-elles conformes au contrat ? | Immédiate | Garde-fou : un KPI de dérive calculé sur des données cassées n'a aucun sens. |
| **Dérive des données** (*data drift*) | La population scorée ressemble-t-elle encore à celle d'entraînement ? | Immédiate | Indicateur **avancé** : un changement de mix annonce souvent un changement de comportement. |
| **Dérive des prédictions** | La distribution des scores bouge-t-elle ? | Immédiate | Indicateur avancé synthétique : le modèle « voit » déjà quelque chose. |
| **Performance** | Le modèle prédit-il encore correctement ? | **Différée** (la cible n'est connue qu'après l'horizon : 12 mois ici) | Indicateur **retardé** mais seul juge de paix. |

La stratégie est donc : **agir sur les indicateurs avancés, confirmer par les indicateurs
retardés**. Un ALM-iste n'a pas le luxe d'attendre 12 mois pour apprendre que son
écoulement projeté était faux.

### 1.2 Référence et fenêtre courante

* **Référence** : l'échantillon de développement du modèle (cohortes 2013-2015), avec des
  scores **hors-pli** (validation croisée) pour que la performance de référence ne soit
  pas optimiste. Elle est figée ; elle ne change qu'à la ré-estimation du modèle.
* **Fenêtre courante** : chaque mois d'émission de 2016-2017, scoré par le modèle figé.
  Chaque fenêtre produit un rapport ; l'historique des rapports est la matière du
  dashboard.
* **Taille minimale** (`min_sample_size: 1000`) : en dessous, le KPI n'est pas calculé
  plutôt que de produire un signal bruité. Avec 20 000 à 45 000 prêts par mois, ce seuil
  est très confortable ; il devient déterminant pour des segments fins.

### 1.3 Échelle de gravité

Trois niveaux, communs à tous les KPIs : `ok`, `warning` (à surveiller, à documenter),
`critical` (action requise : analyse, recalibrage ou ré-estimation). Un quatrième
niveau, `error`, signale qu'un KPI **n'a pas pu être calculé** (données manquantes,
colonne absente) : c'est volontairement traité comme un signal, pas comme un silence.

## 2. Qualité des données en entrée

**Disponibilité des données (`data_availability`).** Avant tout KPI, chaque fenêtre
temporelle doit contenir au moins `min_sample_size` lignes (1 000 pour Lending Club). Une
fenêtre vide ou trop petite ne produit pas de KPI (ils seraient instables) mais **une ligne
`data_availability` en gravité `error`**, avec l'effectif observé et le minimum requis.
Le principe : une absence de données est une erreur opérationnelle à traiter, jamais une
absence de dérive ; sauter silencieusement le mois laisserait un trou invisible dans le
dashboard et un scénario vert.

**Couverture des labels (`label_coverage`).** Les KPI de performance ne portent que sur
les lignes dont la cible et la prédiction sont connues. Si 40 % des labels manquent, un
AUC calculé sur les 60 % restants est une estimation sur un sous-échantillon qui n'est
pas tiré au hasard (prêts plus anciens, plus mûrs). `label_coverage` = part des lignes
labellisées, seuils 0,95 / 0,80 ; chaque KPI de performance répète l'assiette dans ses
détails (`n_total`, `n_labelled`, `excluded_rate`). Pour une cible différée, une fenêtre
récente a une couverture nulle : CRITICAL, c'est-à-dire « performance non encore
mesurable », ce qui est l'information juste. Une couverture faible n'est pas en soi une
preuve de biais : il faut vérifier si les labels manquants dépendent du profil des prêts
ou de la période (mécanisme MCAR / MNAR), ce que fait le contrôle des manquants de l'audit
de qualité (`data_audit.check_missing`) ; c'est la lecture conjointe des deux qui dit si
la performance mesurée est représentative.

**Découpage temporel.** Les fenêtres mensuelles sont construites sur des positions de
lignes et non sur les étiquettes de l'index pandas : un index dupliqué (fréquent après un
`concat`) ferait sinon compter des observations plusieurs fois. Les horodatages non
analysables sont écartés et leur nombre est conservé dans les métadonnées de chaque
rapport (`unparsed_timestamp_rows`).


| KPI | Définition | Seuils (warning / critical) | Justification |
|---|---|---|---|
| `schema_conformity` | Nombre de violations du contrat : colonne absente, type incompatible | — / 1 | Toute violation invalide le reste du rapport. Pas de niveau intermédiaire. |
| `missing_rate` | Part de valeurs manquantes par feature | 0,10 / 0,20 (défaut 0,05 / 0,20) | `emp_length_years` a 6 % de manquants structurels en référence : on alerte sur une **hausse**, pas sur le niveau. Un pic de manquants est le symptôme le plus fréquent d'un incident amont (changement de format, flux partiel). |
| `out_of_range_rate` | Part de valeurs hors de l'intervalle [q0,001 ; q0,999] de la référence | 0,01 / 0,05 | Détecte des changements d'unité (taux en % vs en fraction), des valeurs aberrantes, des extensions de l'offre (montants plus élevés). Les quantiles extrêmes tolèrent les queues légitimes. |
| `unseen_categories_rate` | Part de lignes dont la modalité n'existait pas en référence | 0,01 / 0,05 | Une nouvelle modalité (`application_type = Joint App`, nouveau `purpose`) est invisible pour un modèle entraîné sans elle : l'encodeur la traite comme « inconnue ». |

## 3. Dérive des données

Tous les indicateurs de dérive comparent, feature par feature, la distribution courante à
la distribution de référence. Plusieurs mesures sont conservées car elles n'ont pas la
même sensibilité ni la même lisibilité.

| KPI | Définition | Seuils | Pourquoi celui-là |
|---|---|---|---|
| `psi` (Population Stability Index) | Σ (p_cur − p_ref) · ln(p_cur / p_ref) sur 10 classes de quantiles de la référence (ou une classe par modalité) | 0,10 / 0,25 | Standard de place en banque (scoring, Bâle) : les seuils 0,10 / 0,25 sont connus des validateurs et des auditeurs. Classes par **quantiles de la référence**, donc robuste aux queues lourdes (montants, revenus). |
| `js_distance` (Jensen-Shannon) | Distance bornée dans [0, 1] entre les deux distributions | 0,10 / 0,20 | Bornée et symétrique, elle évite l'explosion du PSI quand une classe se vide (le PSI de `int_rate` atteint 1,3 en septembre 2017 ; la JS plafonne à 0,30 et reste lisible). |
| `ks_test` (Kolmogorov-Smirnov) | Écart maximal entre fonctions de répartition, variables numériques | 0,10 / 0,20 | Sans discrétisation ; sensible aux décalages de position. La **statistique** est seuillée, pas la p-value : avec 40 000 lignes, toute différence est « significative ». Restreint aux variables de tarification et de risque pour garder un rapport lisible. |
| `cramers_v` | Association (χ², corrigée du biais) entre appartenance à la fenêtre et la modalité | 0,10 / 0,30 | Lecture directe pour les catégorielles : « la fenêtre explique-t-elle la modalité ? ». |

**Lecture ALM.** Sur Lending Club, la dérive dominante porte sur `int_rate` (changements
de grille tarifaire de la plateforme) et `application_type` (demandes conjointes
introduites fin 2016). Pour un modèle de remboursement anticipé, une dérive du taux
contractuel est précisément ce qui change l'incitation à refinancer : c'est une dérive
**qu'il faut** relier au comportement, pas seulement constater.

**Ce que la dérive univariée ne voit pas.** Un changement de **relation** entre deux
features (par exemple taux × grade) à distributions marginales stables. C'est le rôle du
*data drift* natif de Dataiku (modèle de dérive multivarié, voir le guide
d'implémentation) et de la validation adversariale de l'audit de fuite.

**Cas limite : variable constante dans la référence.** Les bornes par quantiles
s'effondrent alors en une seule classe et le PSI vaudrait 0 quel que soit le courant — une
variable figée à l'entraînement qui se met à bouger en production serait invisible. La
valeur de référence est donc isolée dans sa propre classe, encadrée de deux classes
« au-dessous / au-dessus », dans les deux backends (pandas et PySpark) ; un courant décalé
déclenche alors un PSI critique. Détecté par les tests de cas limites, corrigé.

## 4. Dérive des prédictions

| KPI | Définition | Seuils | Justification |
|---|---|---|---|
| `prediction_drift` | PSI de la distribution des scores (classes = déciles des scores de référence) ; la moyenne prédite est conservée en détail | 0,10 / 0,25 | Premier signal disponible quand la cible est différée : en septembre-octobre 2016, le PSI des scores monte à 0,07-0,08 et la moyenne prédite passe de 13,0 % à 14,9 % **avant** qu'aucun remboursement anticipé de ces cohortes ne soit observable. |

## 5. Performance

Les métriques de performance exigent la cible. Elles sont donc calculées **avec retard**
(12 mois) et, en production, sur des fenêtres plus anciennes que les indicateurs avancés.

| KPI | Définition | Seuils (Lending Club) | Justification |
|---|---|---|---|
| `auc` | Aire sous la courbe ROC | 0,61 / 0,58 | Mesure le **classement**. Seuils ancrés sur la performance hors temps du champion (0,640) : un comportement de remboursement anticipé reste peu prévisible, des seuils génériques « 0,75 » n'auraient aucun sens. Un AUC stable **n'est pas** une garantie : voir ci-dessous. |
| `brier_score` | Erreur quadratique moyenne des probabilités | 0,135 / 0,145 | Combine classement et calibration ; référence hors temps 0,127. |
| `calibration_error` (ECE) | Écart moyen, pondéré par l'effectif, entre probabilité prédite et fréquence observée sur 10 classes de probabilité de largeur égale ([0, 0,1[, [0,1, 0,2[, …) | 0,02 / 0,04 | **Décisif en ALM** : l'écoulement projeté dépend du niveau prédit, pas du classement. L'ECE triple entre janvier 2016 (0,009) et 2017 (0,03) alors que l'AUC ne bouge pas. |
| `level_bias` | \|moyenne prédite − taux observé\| (signe conservé en détail) | 0,02 / 0,04 | Lecture directe pour le métier : « le modèle sous-estime les remboursements anticipés de 3,3 points ». En projection de flux, 3 points sur un taux de 17 % représentent un cinquième de l'écoulement anticipé. |
| `performance_degradation` | (AUC_ref − AUC_cur) / AUC_ref | 0,05 / 0,10 | Dégradation **relative** du classement, indépendante du niveau absolu ; reste à `ok` sur toute la période, ce qui illustre pourquoi il ne peut pas être le seul indicateur. |
| `log_loss` | Entropie croisée | désactivé | Redondant avec Brier pour la restitution métier ; conservé dans le catalogue. |
| `rmse`, `mae`, `r2` | Régression | configurables | Pour les modèles ALM à cible continue (taux d'écoulement, durée). |

**Pourquoi la calibration prime.** Un modèle de remboursement anticipé n'est pas utilisé
pour trier des prêts mais pour **projeter** combien de capital sera remboursé avant terme.
Un modèle qui classe bien mais prédit 14 % quand on observe 17 % fausse directement le
gap de taux et l'écoulement des actifs. C'est pourquoi la règle de sélection du
benchmark (section 7) filtre d'abord sur l'ECE.

## 6. Définition de la cible et audit de fuite

### 6.1 Cible à horizon fixe

`prepaid_12m = 1` si le prêt est intégralement remboursé (*Fully Paid*) au plus tard 12
mois après son émission **et** au moins un mois avant son terme contractuel.

Pourquoi un horizon : sans lui, la cible « remboursé avant terme » exclut les prêts
encore en cours, et les cohortes récentes paraissent prépayer à 84 % (biais de censure,
notebook 01). L'horizon fixe rend toutes les cohortes comparables ; en contrepartie, une
cohorte n'entre dans le périmètre que si son horizon est écoulé à la date d'arrêté
(cohortes jusqu'à décembre 2017 pour un fichier arrêté en mars 2019).

### 6.2 Contrôles de fuite (`alm_monitoring.leakage`)

1. **Liste noire** : 23 colonnes post-octroi (`total_pymnt`, `last_pymnt_d`,
   `out_prncp`, `recoveries`…) ne peuvent jamais être des features. Un modèle qui les
   utilise affiche une AUC proche de 1 et ne vaut rien en production.
2. **Découpe temporelle** : la dernière date de la référence précède strictement la
   première de la surveillance.
3. **Pouvoir discriminant unitaire** : aucune feature isolée ne doit séparer la cible
   avec une AUC > 0,90 (la meilleure, `revol_util`, est à 0,56).
4. **Validation adversariale** : un classifieur tente de distinguer référence et
   surveillance. AUC observée 0,92 : les fenêtres sont très distinguables, mais aucune
   feature post-octroi n'est présente et aucune feature n'est discriminante seule. La
   lecture est donc **dérive multivariée** (grille tarifaire × grade × FICO), pas fuite.
   Ce contrôle ne dépasse jamais `warning` à lui seul : il désigne ce qu'il faut examiner.

### 6.3 Exclusion des attributs opérationnels

`initial_list_status` (mode de listage du prêt sur la place de marché) est exclu des
features : il décrit la plateforme, pas l'emprunteur, dérive au gré de la politique
interne et dominait la validation adversariale. En banque, l'équivalent serait un code
de canal de distribution ou de campagne.

## 6 bis. Qualité des données et ground truth (`alm_monitoring.data_audit`)

L'audit répond, avant tout entraînement, à « ces données sont-elles exploitables ? » par dix
contrôles gradués et un verdict. Sur le fichier complet (2 260 668 prêts, empreinte SHA-256
`55c16f75…`), verdict **exploitable avec avertissements** :

| Contrôle | Constat sur Lending Club | Décision |
|---|---|---|
| Doublons | 0 ligne identique, 0 identifiant répété, 44 collisions sur 7 champs (0,002 %) | Aucune suppression : collisions plausibles |
| Règles métier | 0,6 % de valeurs impossibles : `revol_util` > 100 % (7 343), DTI > 100 (2 563), revenu nul (1 667), revenu > 5 M$ (93), mensualité incohérente (2 106) | **NaN, pas suppression ni plafonnement** : supprimer créerait un biais de sélection, plafonner inventerait une valeur ; les modèles retenus tolèrent les manquants |
| Manquants | `emp_length_years` 6 %, `mort_acc` 2,7 %. Le manquant de `mort_acc` est **informatif** : taux de remboursement anticipé 10,3 % quand il manque, 14,9 % sinon | Indicateur `mort_acc_missing` ajouté aux features ; pas d'imputation silencieuse |
| Aberrants robustes | `revol_bal` 2,7 % au-delà de 5 MAD, `annual_inc` jusqu'à 61 M$ | Couverts par les règles métier et la robustesse des arbres |
| Distributions | 6 variables très asymétriques (`annual_inc`, `revol_bal`, `dti`…) | Sans effet sur le boosting ; log/quantiles pour la régression logistique |
| Cardinalité | Aucune colonne constante ni à forte cardinalité | — |
| Multicolinéarité | VIF ≈ 50 sur `loan_amnt` / `installment` (la mensualité est une fonction d'annuité du montant, du taux et de la durée) | Attendu ; à traiter pour la logistique (une des deux) |
| Cohérence temporelle | 130 mois sans trou ; 36 mois sous 1 000 prêts, tous avant 2012 | Référence et surveillance commencent en 2013 |
| Cible | 14,8 % de positifs (8 % à 17 % selon l'année) ; 2 749 statuts ambigus (« Does not meet the credit policy ») traités comme leurs homologues | Déséquilibre **modéré** : voir ci-dessous |

### Déséquilibre des classes : impact et traitement

Un positif pour six négatifs est un déséquilibre modéré. Il ne biaise pas un modèle qui
optimise la log-vraisemblance : il réduit la quantité de signal sur la classe rare, donc le
plafond d'AUC. Le vrai risque est la dérive du taux de base dans le temps (13 % → 17 %), que
`level_bias` surveille. Décisions :

* **pas de rééquilibrage** (SMOTE, sous-échantillonnage, `class_weight`) : ces techniques
  déforment les probabilités, donc la calibration, critère n° 1 en ALM. Un modèle rééquilibré
  prédirait 40 % de remboursements anticipés pour 15 % observés ;
* métriques insensibles au déséquilibre (AUC, Brier, ECE) complétées par la **PR-AUC**
  (précision-rappel), sensible à la classe rare : 0,249 hors temps contre 0,156 pour le
  taux de base ;
* taux de base par période documenté dans la fiche modèle.

### Tests statistiques : ce qu'on utilise et ce qu'on écarte

* **KS, chi², Cramér's V** : statistique seuillée, p-value en détail. Avec 40 000 lignes par
  mois toute différence est « significative » ; seule l'ampleur compte.
* **Test t, test de normalité** : écartés comme indicateurs (aucun modèle retenu ne suppose
  la normalité ; la significativité est acquise d'office). Les comparaisons de moyennes
  passent par des tailles d'effet quand elles sont utiles.
* **A/B test** : sans objet (pas d'intervention sur des clients). Son équivalent pour un
  modèle est le **champion-challenger** (§ 7 bis).

## 7 bis. Feature engineering, explicabilité et champion-challenger

### Features métier

Sept features dérivées, toutes connues à l'émission, évaluées **à protocole identique**
(même référence, même fenêtre hors temps) avant d'être retenues :

| Feature | Définition | Pourquoi |
|---|---|---|
| `rate_spread` | taux du prêt − taux moyen des prêts émis le même mois | Incitation à refinancer : moteur économique du remboursement anticipé. 2ᵉ variable la plus importante du champion |
| `loan_to_income`, `installment_to_income` | taux d'effort | Lisibles par un analyste crédit |
| `credit_history_years` | ancienneté du premier compte de crédit | Maturité financière |
| `mort_acc_missing` | indicateur du manquant informatif | Issu de l'audit |
| `term_x_rate` | durée × taux | Coût total du crédit |
| `fico_band` | classes FICO usuelles | Robustesse, lisibilité |

Résultat hors temps (2016) : le gradient boosting passe de Brier 0,1275 à **0,1273** et
d'AUC 0,639 à 0,641 ; la régression logistique gagne surtout en calibration (ECE 0,020 →
0,016). Gain réel mais modeste : les features métier améliorent d'abord la **lisibilité** et
la **calibration**, ce qui est le bon ordre de priorité en ALM. Tout ce qui a été testé est
conservé dans `reports/benchmark_lending_club.json`, y compris ce qui n'apporte rien.

### Explicabilité (`modeling/explain.py`)

* **Importance par permutation**, mesurée hors temps sur la dégradation du score de Brier :
  `total_acc`, `rate_spread`, `revol_util`, `open_acc`, `fico` en tête.
* **Dépendance partielle** et **vérification du sens économique** : la probabilité de
  remboursement anticipé croît avec le taux du prêt, avec l'écart au marché et avec le FICO
  (les bons profils refinancent plus facilement). Les trois signes attendus sont conformes ;
  un signe inverse bloquerait la mise en production.

### Champion-challenger (`modeling/challenger.py`)

Équivalent, pour un modèle comportemental, du test A/B : le challenger tourne en parallèle du
champion sur les 24 fenêtres mensuelles de surveillance, et une règle écrite à l'avance décide :

* **promouvoir** si le Brier est meilleur sur au moins deux tiers des fenêtres et la
  calibration n'est pas dégradée en moyenne ;
* **rejeter** s'il est moins bon sur la majorité des fenêtres ;
* **indéterminé** sinon : prolonger l'observation.

Appliqué au meilleur modèle sans features métier contre le champion : rejeté (Brier dégradé
sur 79 % des fenêtres). La même procédure servira à toute ré-estimation future.

### Dérive des données et dérive de concept

Les deux se distinguent et n'appellent pas la même réponse :

| | Ce qui change | Signal | Réponse |
|---|---|---|---|
| Dérive des données | la distribution des entrées | PSI, JS, KS, Cramér's V, drift natif du MES | Documenter ; surveiller la performance ; ré-estimer si le support sort de la référence |
| Dérive de concept | la relation entrées → cible | calibration, biais de niveau, Brier à AUC stable | **Recalibrer** si seul le niveau bouge ; **ré-estimer** si le classement se dégrade aussi |

Sur 2016-2017, les deux coexistent : dérive des données sur `int_rate` et `application_type`,
dérive de concept visible dans le biais de niveau (−1 → −3,5 points à AUC constante).

### Reproductibilité

Graines fixées, empreinte SHA-256 du fichier source dans l'audit, versions des librairies
figées par `uv.lock`, fiche modèle horodatée avec la règle de sélection : chaque chiffre de
ce document est rattaché à un fichier, un code et une date.

## 7. Benchmark et règle de sélection

Trois algorithmes × deux jeux de features, entraînés **uniquement** sur la référence,
évalués hors-pli (développement) puis **hors temps** sur 2016 :

| Modèle | AUC dév. | AUC hors temps | PR-AUC | Brier | ECE | Biais de niveau |
|---|---|---|---|---|---|---|
| Taux de base (plancher) | 0,500 | 0,500 | 0,156 | 0,1321 | 0,022 | −2,2 pt |
| Régression logistique, features de base | 0,627 | 0,624 | 0,230 | 0,1286 | 0,020 | −2,0 pt |
| Régression logistique, features métier | 0,636 | 0,629 | 0,234 | 0,1281 | 0,016 | −1,5 pt |
| HistGradientBoosting, features de base | 0,640 | 0,639 | 0,247 | 0,1275 | 0,020 | −2,0 pt |
| **HistGradientBoosting, features métier** | 0,645 | 0,641 | 0,249 | **0,1273** | 0,019 | −1,9 pt |

Règle : **champion = meilleur Brier hors temps parmi les candidats dont l'ECE ≤ 0,025** ;
à défaut, le mieux calibré. Le modèle naïf n'est jamais champion. La tolérance de 0,025
(et non 0,02) évite qu'un arrondi départage des modèles équivalents.

Lecture : le gain du gradient boosting est réel mais modeste (+0,017 d'AUC) ; la
régression logistique avec features métier resterait défendable si la lisibilité des
coefficients primait pour la validation interne (et elle est la mieux calibrée). Tous
sous-estiment déjà le niveau de 2016 de 1,5 à 2 points : la dérive comportementale est
antérieure à la mise en production, ce que le monitoring confirme mois après mois.

## 8. Ce que le dispositif ne couvre pas (et ce qui est prévu)

* **Segments** : les KPIs sont calculés sur la population entière ; une dérive concentrée
  sur un segment (prêts 60 mois, grade E) peut être diluée. Prévu : KPIs par segment avec
  agrégation hiérarchique des gravités.
* **Référence glissante** : la référence est figée jusqu'à ré-estimation. Prévu : fenêtre
  de référence à décroissance temporelle après recalibrage validé.
* **Incertitude** : pas d'intervalle de confiance sur les KPIs ; avec 20 000+ lignes par
  mois c'est acceptable, ce ne le serait pas pour des segments fins.
* **Saisonnalité et régimes** : la référence est une période fixe ; les remboursements
  anticipés ont une saisonnalité et dépendent du niveau des taux. Un PSI « élevé » en
  décembre peut être un effet calendaire. Prévu : référence au même mois de l'année
  précédente en complément de la référence d'entraînement, et référence par régime de taux.
* **Interactions** : dérives univariées complétées par le modèle de dérive multivarié natif
  du Model Evaluation Store, mais pas d'analyse des relations entre variables (une dérive
  jointe `int_rate × term` sans dérive marginale passerait inaperçue).
* **Causalité** : le dispositif signale, il n'explique pas. L'analyse (grille tarifaire,
  nouveau produit) reste humaine ; les détails conservés (`detail_*`) sont là pour la
  rendre rapide.
