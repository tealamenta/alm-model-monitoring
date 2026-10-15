# Dashboard Dataiku « Monitoring ALM – Remboursement anticipé »

Le dashboard est construit **sans code**, avec les charts natifs de DSS, sur deux sources :

* le dataset `monitoring_kpis` (table longue : `run_at`, `window_start`, `kpi`,
  `category`, `column`, `value`, `severity`, `warning_threshold`, `critical_threshold`) ;
* les insights publiés depuis le **Model Evaluation Store** (performance et data drift natifs).

Il est pensé pour deux lecteurs : le **métier ALM** (page 1, en 30 secondes : est-ce que le
modèle est encore utilisable ?) et le **data scientist** (pages 2-4, pour diagnostiquer).

## Page 1 – Synthèse (métier)

| Tuile | Source | Construction |
|---|---|---|
| Statut global de la dernière fenêtre | `monitoring_kpis` | Chart « KPI » : max de `severity_rank` (colonne calculée `ok=0, warning=1, critical=2, error=3`) filtré sur le dernier `run_id` ; couleur conditionnelle vert / orange / rouge. |
| Nombre de dépassements | `monitoring_kpis` | Chart « KPI » : count où `severity ≠ ok`, dernier `run_id`. |
| Biais de niveau | `monitoring_kpis` | Chart « Lines » : `detail_signed_bias` par `window_start` pour `kpi = level_bias`, lignes de référence à ±0,02 et ±0,04. C'est **le** graphique ALM : au-dessus de zéro le modèle surestime l'écoulement, en dessous il le sous-estime. |
| Calibration et AUC | `monitoring_kpis` | Chart « Lines », deux axes : `value` pour `kpi ∈ {auc, calibration_error}` par `window_start`. Montre un classement stable et une calibration qui se dégrade. |
| Heatmap de gravité | `monitoring_kpis` | Chart « Pivot table » : lignes = `kpi`·`column`, colonnes = `window_start`, cellule = `severity_rank` max, échelle de couleur discrète. |

Filtre global de page : `model_name`, `model_version`.

## Page 2 – Dérive des données

* « Lines » : `value` du `psi` par `window_start`, une courbe par `column`, lignes de seuil
  0,10 / 0,25. Légende limitée aux 8 features les plus dérivantes (filtre `value > 0.05`).
* « Bars » : PSI de la dernière fenêtre par `column`, trié décroissant, couleur = `severity`.
* Insight MES « Input data drift » : précision du modèle de dérive et importance des
  features, pour confronter le drift natif (multivarié) aux PSI univariés.
* Tableau : `ks_test` et `cramers_v` de la dernière fenêtre avec leurs p-values
  (`detail_p_value`).

## Page 3 – Dérive des prédictions et performance

* « Lines » : `prediction_drift` et `detail_mean_current` (moyenne prédite) par fenêtre ;
  lecture : la moyenne prédite monte avant que la cible soit observable.
* Insight MES « Performance » de la dernière évaluation (AUC, log-loss, matrice).
* « Lines » : `brier_score`, `log_loss`, `performance_degradation` par fenêtre.
* Tableau `monitoring_kpis` filtré `category = performance`, avec seuils.

## Page 4 – Qualité des données

* « Bars » : `missing_rate` par `column`, dernière fenêtre, seuils en lignes.
* « Lines » : `out_of_range_rate` et `unseen_categories_rate` par fenêtre et par colonne.
* Chart « KPI » : nombre de violations de schéma (`kpi = schema_conformity`, `value`).
* Insight « Metrics & checks » du dataset `lc_current` (volume, fraîcheur).

## Conventions

* Couleurs de gravité uniformes sur toutes les pages : `ok` vert, `warning` orange,
  `critical` rouge, `error` gris.
* Chaque graphique porte en sous-titre la **définition du KPI** (reprise de
  `docs/methodologie_kpi.md`) et son seuil, pour que la lecture ne dépende pas d'un expert.
* Le dashboard est exporté en PDF à chaque exécution du scénario (étape « Export
  dashboard ») et joint au mail d'alerte.

## Équivalent hors Dataiku

`alm_monitoring.viz` produit les mêmes figures en Plotly (`reports/figures/*.html`) à
partir de `reports/<modèle>/history.csv` ; le notebook `03_kpis_et_dashboard` les assemble.
