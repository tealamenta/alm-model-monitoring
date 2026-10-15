# Intégration Dataiku DSS

Ce dossier contient tout ce qu'il faut pour déployer le framework dans un projet Dataiku,
en s'appuyant sur les **fonctionnalités natives** (Model Evaluation Store, scénarios,
métriques & checks, reporters, plugins, dashboards) et en y ajoutant les KPIs ALM
spécifiques du package `alm_monitoring`.

> **Exécuté et validé sur DSS 15.1 Free Edition** (octobre 2026) : Flow, recette,
> règle Data Quality, Saved Model MLflow, Model Evaluation Store, scénario et dashboard.
> Les notebooks réellement exécutés sont dans [`notebooks/`](notebooks/), l'export du
> projet (sans données) dans [`export/`](export/), les captures dans
> [`../docs/captures/`](../docs/captures/) et le retour d'expérience (écarts avec la
> documentation, corrections) au §6 du
> [guide d'implémentation](../docs/guide_implementation_dataiku.md).

## Architecture dans DSS

```
                 ┌──────────────────────────────────────────────────────────────────┐
                 │  Flow                                                            │
 reference ─────►│  ┌─────────────┐    ┌──────────────────┐                        │
 (dataset)       │  │ Saved Model │───►│ Evaluation recipe│───► Model Evaluation   │
 current  ──────►│  │ (MLflow)    │    │  (native)        │     Store (natif :     │
 (dataset)       │  └─────────────┘    └──────────────────┘     perf + data drift) │
       │         │                                                                  │
       └────────►│  ┌──────────────────────────┐                                    │
                 │  │ Recette Python           │───► monitoring_kpis (dataset)      │
                 │  │ alm_monitoring (KPIs ALM)│       │  métriques & checks        │
                 │  └──────────────────────────┘       │  dashboard                 │
                 └──────────────────────────────────────┼───────────────────────────┘
                                                        ▼
                 Scénario mensuel : build MES → build monitoring_kpis → run checks
                 → variables de scénario → reporter (mail / Teams) si WARNING+
```

Les deux voies sont complémentaires :

| Besoin | Natif Dataiku | Ajout `alm_monitoring` |
|---|---|---|
| Métriques de performance standard (AUC, log-loss…) | Model Evaluation Store | — |
| Data drift global (modèle de dérive) et univarié | Model Evaluation Store, onglet *Input data drift* | PSI / JS / KS / Cramér's V par feature, **seuillés et historisés** |
| Dérive des prédictions | — (partiel) | `prediction_drift` |
| Calibration, biais de niveau (décisifs en ALM) | — | `calibration_error`, `level_bias` |
| Qualité des données en entrée | Métriques & checks sur dataset | `missing_rate`, `out_of_range_rate`, `unseen_categories_rate`, `schema_conformity` |
| Seuils justifiés par modèle | — | YAML par modèle (`configs/`) |
| Alertes | Reporters de scénario | Check Python + variables `${alm_overall_severity}` |
| Restitution | Dashboard sur `monitoring_kpis` + insights du MES | Figures Plotly exportables |

## Contenu

| Fichier | Rôle |
|---|---|
| `recipes/00_setup_project.py` | Script d'amorçage (hors DSS avec `dataikuapi`, ou dans un notebook DSS sans clé d'API) : importe le champion MLflow comme *Saved Model*, l'évalue sur la référence, crée le Model Evaluation Store et la recette d'évaluation, construit le MES et lit ses métriques. |
| `notebooks/00_setup_mes.py` | Version notebook **exécutée sur DSS 15.1** : typage du schéma, import MLflow v2 avec `features_list`, évaluation de référence, MES, recette d'évaluation, build. |
| `notebooks/01_setup_scenario.py` | Création du scénario step-based par l'API, **exécutée sur DSS 15.1**. |
| `export/ALM_MONITORING.zip` | Export du projet DSS sans données, importable sur toute instance (Projet → Import). |
| `recipes/01_compute_alm_kpis.py` | Recette Python (à coller dans une recette du Flow) : calcule les KPIs ALM entre `reference` et `current` et alimente `monitoring_kpis`. |
| `recipes/03_audit_qualite_donnees.py` | Recette Python : audit de qualité des données (10 contrôles) → `audit_constats`, `audit_tables`, variable de scénario `alm_data_exploitable`. |
| `recipes/04_preparation_et_features.py` | Recette Python : nettoyage, cible à horizon fixe, features métier, découpe référence / surveillance ; même code qu'à l'entraînement (pas d'écart train/serve). |
| `recipes/02_scenario_monitoring.py` | Scénario *custom Python* : enchaîne build du MES, build des KPIs, fusion des deux sources, checks, variables de scénario et message sur un canal (mail/Teams). |
| `checks/custom_check_monitoring_kpis.py` | Check Python personnalisé sur `monitoring_kpis` : échoue si la dernière exécution est CRITICAL, avertit si WARNING. |
| `scenarios/alm_monitoring_scenario.json` | Définition d'un scénario *step-based* équivalent (trigger mensuel, étapes, reporter), importable ou à reproduire dans l'interface. |
| `plugin/alm-monitoring-plugin/` | Plugin DSS : recette visuelle « ALM Monitoring KPIs » avec formulaire de paramètres, pour qu'un utilisateur métier monitore un nouveau modèle sans coder. |
| `dashboard/dashboard_monitoring.md` | Spécification du dashboard Dataiku (tuiles, graphiques, filtres) construit sur `monitoring_kpis` et sur les insights du MES. |

## Pré-requis

* Code env Python 3.11+ avec `mlflow`, `scikit-learn` (même version que l'entraînement), `cloudpickle` et `statsmodels` (requis par le drift natif) ; le package `alm-model-monitoring` installé dans ce code env **ou** dans le builtin (`$DSS_HOME/bin/pip install .`).
* Le package installé dans le code env **ou** copié dans `lib/python` du projet.
* Datasets `lc_reference` et `lc_current` (sorties de `scripts/train_lending_club.py`,
  importables par upload ou connexion SQL).
* Un canal de messagerie configuré (Administration → Settings → Messaging channels)
  pour les alertes.
