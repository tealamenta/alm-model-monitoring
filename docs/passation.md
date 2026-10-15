# Fiche de passation

Ce document répond au troisième axe de la mission (« documentation et capitalisation,
passation de compétences ») : ce qu'une personne reprenant le dispositif doit savoir pour
l'exploiter, le maintenir et l'étendre, en une lecture.

## 1. En trois phrases

Le framework compare une **fenêtre de référence** (données d'entraînement scorées) à une
**fenêtre courante** et produit une table de KPIs seuillés (dérive, prédictions,
performance, qualité). Il est piloté par un fichier **YAML par modèle** et s'exécute en
ligne de commande, en bibliothèque, dans une recette Dataiku ou via un plugin. Les
résultats alimentent un dataset `monitoring_kpis` que les checks, le scénario et le
dashboard exploitent.

## 2. Parcours de prise en main (une demi-journée)

| Étape | Commande / lecture | Durée |
|---|---|---|
| Installer | `uv sync --all-extras --group dev` (ou `pip install -e ".[viz,spark,mlflow]"`) | 10 min |
| Comprendre les données | `notebooks/01_comprendre_les_donnees.ipynb` | 30 min |
| Comprendre le modèle | `notebooks/02_benchmark_et_leakage.ipynb` + `docs/methodologie_kpi.md` §6-7 | 45 min |
| Comprendre les KPIs | `docs/methodologie_kpi.md` §1-5, `alm-monitor list-kpis` | 45 min |
| Voir le monitoring | `notebooks/03_kpis_et_dashboard.ipynb`, `reports/lending_club/figures/` | 30 min |
| Rejouer le pipeline | `make demo` (échantillon, 3 min) puis `make pipeline` (complet, 10 min) | 15 min |
| Dataiku | `docs/guide_implementation_dataiku.md`, `dataiku/README.md` | 45 min |

## 3. Exploitation courante

### Chaque mois (ou à chaque nouvelle fenêtre)

1. S'assurer que `current.csv` (ou `lc_current`) contient la nouvelle cohorte scorée.
2. `python scripts/run_monitoring.py` (ou le scénario Dataiku).
3. Lire `reports/lending_club/summary.md` : statut par fenêtre et KPIs les plus souvent
   en dépassement.
4. Si `warning` : documenter dans le journal de suivi (quelle feature, quelle ampleur,
   hypothèse d'explication). Si `critical` : ouvrir une analyse ; décider recalibrage
   (biais de niveau seul) ou ré-estimation (dérive des données + dégradation).

### Grille de décision indicative

| Signal | Lecture | Action |
|---|---|---|
| `missing_rate` / `schema_conformity` en dépassement | Incident de données amont | Bloquer la diffusion des projections, corriger le flux, rejouer |
| `psi` élevé sur une feature, performance stable | Changement de mix, modèle robuste | Documenter ; surveiller la calibration les mois suivants |
| `prediction_drift` en hausse, cible pas encore connue | Le modèle anticipe un changement | Prévenir le métier ; préparer un scénario alternatif dans les projections |
| `level_bias` / `calibration_error` critiques, `auc` stable | Le classement tient, le niveau est faux | **Recalibrage** (ajustement du niveau, isotonic/Platt) sans ré-estimer |
| `auc` et `performance_degradation` critiques | Le modèle ne discrimine plus | **Ré-estimation** complète, nouvelle référence, nouvelle version |

## 4. Maintenance

### Ajouter un modèle

Écrire `configs/<modele>.yaml` (copier `lending_club_prepayment.yaml`), valider
(`alm-monitor validate-config`), fournir `reference.csv` / `current.csv`, lancer. Aucun
code à modifier. Dans Dataiku : dupliquer recette + scénario + dashboard, ou utiliser le
plugin.

### Ajouter un KPI

1. Créer une classe dans `kpis/` héritant de `KPI` (ou `_PerformanceKPI`,
   `_BinnedKPI`), décorée `@register`, avec `name`, `category`, `description`,
   `higher_is_worse`, `default_thresholds`, `compute(ctx)`.
2. Écrire ses tests dans `tests/unit/test_kpis.py`.
3. L'activer dans les YAML concernés. Il apparaît automatiquement dans `list-kpis`, les
   rapports, les figures, la recette et le plugin.

### Changer un seuil

Dans le YAML du modèle, avec un commentaire justifiant la valeur (c'est une exigence de
traçabilité, pas une coquetterie). Ne jamais modifier les défauts dans le code pour un
seul modèle.

### Ré-estimer le modèle

`python scripts/train_lending_club.py` → nouveau champion, nouvelle `reference.csv`
(scores hors-pli), `model_card.json` mis à jour. Incrémenter `model.version` dans le YAML.
Exporter en MLflow et importer la nouvelle version dans le Saved Model Dataiku.

## 5. Vérifier que tout va bien

```bash
make check     # ruff + mypy + bandit + pip-audit + pytest (couverture ≥ 90 %)
make test      # tests seuls
pytest -m "not slow"   # sans le test PySpark
```

## 6. Pièges connus

* **Fuite de données** : ne jamais ajouter une colonne de `POST_ORIGINATION_COLUMNS`
  aux features ; l'audit refuse l'entraînement (`train_lending_club.py` sort en code 2).
* **Biais de censure** : la cible est à horizon fixe (12 mois) ; une cohorte dont
  l'horizon n'est pas écoulé est exclue automatiquement. Ne pas « forcer » les cohortes
  récentes.
* **Validation adversariale à 0,9** : ce n'est pas une fuite en soi (voir méthodologie
  §6.2) ; vérifier que la liste noire et le pouvoir discriminant unitaire sont verts.
* **Fatigue d'alerte** : si tous les mois sont `critical`, ce sont les seuils qui sont
  faux, pas le modèle. Réancrer les seuils de performance sur la performance hors temps.
* **Spark et NaN** : Spark distingue NULL et NaN ; le backend PySpark les traite tous deux
  comme manquants pour rester aligné sur pandas.

## 7. Contacts et ressources

* Documentation API Dataiku : `developer.dataiku.com` (Model Evaluation Stores,
  Scenarios, Plugins).
* Références PSI / stabilité : pratiques de validation de modèles de scoring (seuils
  0,10 / 0,25).
* Historique des décisions : `docs/methodologie_kpi.md` (cible, exclusions, règle de
  sélection) et les commentaires des YAML.
