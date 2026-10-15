# Guide d'implémentation sur Dataiku DSS

Ce guide répond au deuxième axe de la mission (« implémentation technique et
industrialisation sur Dataiku ») : comment déployer le dispositif de monitoring dans un
projet DSS en exploitant les fonctionnalités natives, et comment le rendre applicable à
tout nouveau modèle sans développement supplémentaire.

> **Validé sur une instance.** Le dispositif a été exécuté de bout en bout sur Dataiku DSS
> **15.1 Free Edition** (macOS, Python 3.12) : recette Python → `monitoring_kpis`
> (3 144 lignes), métriques et règle Data Quality (ERROR, 33 dépassements), Saved Model
> importé de MLflow, Model Evaluation Store (AUC 0,642, data drift 0,771), scénario
> step-based et dashboard. Les captures sont dans `docs/captures/`, l'export du projet
> dans `dataiku/export/`, et les écarts constatés entre la documentation et DSS 15.1 sont
> consignés au §6. Les notebooks réellement exécutés sont dans `dataiku/notebooks/`.

## 1. Vue d'ensemble

```
Entraînement (hors DSS ou dans DSS)        Projet DSS « ALM_MONITORING »
─────────────────────────────────         ────────────────────────────────────────────────
train_lending_club.py                     Flow
   │  champion.joblib                       lc_reference ─┐
   ▼                                        lc_current  ──┼─► recette Python ─► monitoring_kpis
export_mlflow_model.py                                    │     (KPIs ALM)         │  métriques & checks
   │  modèle MLflow                                       │                        │  dashboard
   ▼                                        Saved Model ◄─┘                        │
00_setup_project.py  ──────────────────►      (MLflow)  ─► recette d'évaluation ─► Model Evaluation Store
                                                                                   (perf + data drift natifs)
                                          Scénario mensuel
                                            build MES → build monitoring_kpis → métriques → checks
                                            → variables ${alm_*} → reporter mail/Teams si WARNING+
```

Deux sources de vérité, volontairement complémentaires :

* le **Model Evaluation Store** (MES) porte l'évaluation **native** du modèle (métriques
  standard, drift multivarié par modèle de dérive, drift univarié interactif) ;
* le dataset **`monitoring_kpis`** porte les KPIs **ALM** (calibration, biais de niveau,
  PSI seuillés, qualité des données) avec seuils justifiés et historique.

Le scénario lit les deux et produit une synthèse unique.

## 2. Mise en place pas à pas

### 2.1 Code env et package

1. Créer un code env Python 3.11+ (Administration → Code envs) avec :
   `alm-model-monitoring` (ce dépôt, `pip install git+https://…` ou wheel), `mlflow`,
   `scikit-learn`, `pydantic`, `plotly`.
2. Alternative sans packaging : copier `src/alm_monitoring` dans `lib/python` du projet
   et `configs/*.yaml` dans `lib/python/configs`. Les recettes importent
   `alm_monitoring` indifféremment.

### 2.2 Datasets

| Dataset | Contenu | Source |
|---|---|---|
| `lc_reference` | 791 538 prêts 2013-2015 avec features, cible `prepaid_12m`, score hors-pli | `data/processed/lending_club/reference.csv` (upload ou table SQL) |
| `lc_current` | Prêts à surveiller, scorés par le modèle figé | Alimenté chaque mois par le flux d'octroi + recette de scoring |
| `monitoring_kpis` | Table longue des KPIs (sortie de la recette Python, mode *append*) | Créé par la recette |

Activer sur `lc_current` les **métriques natives** (nombre de lignes, fraîcheur, taux de
nulls par colonne) et un **check** « nombre de lignes > 1 000 » : c'est la première ligne
de défense, avant même les KPIs.

### 2.3 Saved Model et Model Evaluation Store (natif)

Exécuter, hors DSS, avec `DSS_HOST` et `DSS_API_KEY` dans l'environnement :

```bash
python scripts/export_mlflow_model.py          # models/mlflow/prepayment_champion
python dataiku/recipes/00_setup_project.py --project ALM_MONITORING \
    --mlflow-model models/mlflow/prepayment_champion \
    --current-dataset lc_current --code-env alm-monitoring
```

Le script :

1. crée (ou réutilise) un *Saved Model* de type MLflow (`create_mlflow_pyfunc_model`),
   importe la version (`import_mlflow_version_from_path`), renseigne cible et classes
   (`set_core_metadata`) et l'active ;
2. crée le **Model Evaluation Store** (`create_model_evaluation_store`) ;
3. crée la **recette d'évaluation** (entrée : saved model + `lc_current`, sortie : MES) ;
4. construit le MES et lit la dernière *Model Evaluation* (`get_latest_model_evaluation`,
   `get_full_info().metrics`) puis le drift natif (`compute_drift()`).

Chaque build du MES ajoute une évaluation horodatée ; l'onglet *Input data drift* permet
d'explorer le drift univarié et l'importance des features dans le modèle de dérive.
Les modèles **entraînés dans DSS** (Visual ML) suivent le même chemin sans import.

### 2.3 bis Audit de qualité des données et préparation

Deux recettes précèdent le monitoring dans le Flow :

* `dataiku/recipes/03_audit_qualite_donnees.py` : les dix contrôles de l'audit
  (`alm_monitoring.data_audit`) sur `lc_raw`, sorties `audit_constats` et `audit_tables`,
  variable de scénario `alm_data_exploitable` qui conditionne l'étape d'entraînement.
* `dataiku/recipes/04_preparation_et_features.py` : nettoyage (valeurs impossibles → NaN),
  cible à horizon fixe, features métier, découpe référence / surveillance. Le même code sert
  à l'entraînement hors DSS : aucun écart train/serve.

Mapping avec les outils natifs, à montrer en entretien :

| Besoin | Natif DSS | Ce que la recette ajoute |
|---|---|---|
| Explorer une colonne (manquants, distribution, aberrants) | **Explore → Analyze** | — (exploration interactive, pas de verdict) |
| Règles simples (lignes, unicité d'`id`, nulls, valeurs attendues) | **Metrics & Checks**, **Data Quality rules** sur `lc_raw` | Règles croisées (mensualité vs annuité), mécanisme des manquants, VIF, qualité de la cible, verdict |
| Tests statistiques, corrélations | **Statistics worksheets** | Automatisation mensuelle, tailles d'effet |
| Transformations lisibles par le métier | **Prepare recipe** | Mêmes formules en code testé, identiques en scoring |
| Doublons | Recette **Distinct**, check d'unicité | Doublons sémantiques |
| Comparer des algorithmes, importances, dépendance partielle | **Visual ML** (Lab) | Validation **hors temps** (Visual ML coupe aléatoirement par défaut), règle de sélection, PR-AUC, champion-challenger sur fenêtres |

### 2.4 Recette Python des KPIs ALM

Créer une recette Python avec entrées `lc_reference`, `lc_current` et sortie
`monitoring_kpis`, et y coller `dataiku/recipes/01_compute_alm_kpis.py`. Paramètres via
les variables du projet (`alm_config_path`, `alm_period`, …). La recette :

* charge la configuration YAML (contrat de données + seuils) ;
* exécute `ModelMonitor.run_over_time` (une fenêtre par mois) ;
* écrit les résultats en *append* dans `monitoring_kpis` ;
* publie les variables de scénario `alm_overall_severity`, `alm_n_breaches`.

Sur de gros volumes, la variante **PySpark** (`alm_monitoring.spark.drift`) calcule les
KPIs de dérive dans une recette PySpark avec les mêmes formules.

### 2.5 Métriques & checks sur `monitoring_kpis`

* Métriques : `max(severity_rank)` et `count(severity = critical)` sur le dernier
  `run_id` (métriques SQL ou Python).
* Check Python personnalisé : `dataiku/checks/custom_check_monitoring_kpis.py` renvoie
  `ERROR` si CRITICAL, `WARNING` si WARNING, avec le top 5 des dépassements dans le
  message. C'est ce check qui fait échouer le scénario et déclenche les reporters.

### 2.6 Scénario

Deux options équivalentes, fournies toutes les deux :

* **Séquence d'étapes** (`dataiku/scenarios/alm_monitoring_scenario.json`) : build du MES,
  build de `monitoring_kpis`, calcul des métriques, exécution des checks, étape Python de
  synthèse, reporter mail conditionné à `WARNING`/`FAILED`. À reproduire dans l'interface
  ou à adapter dans un export de projet.
* **Custom Python** (`dataiku/recipes/02_scenario_monitoring.py`) : même enchaînement
  piloté par l'API `dataiku.scenario.Scenario` (`build_evaluation_store`, `build_dataset`,
  `compute_dataset_metrics`, `run_dataset_checks`, `set_scenario_variables`,
  `get_message_sender`). Plus souple pour fusionner MES et KPIs ALM dans un même message.

Déclencheurs : mensuel (le 2 du mois à 06:00) et « dataset `lc_current` modifié ».

### 2.7 Alertes

* Reporter de scénario (mail, Teams, Slack) conditionné au statut, avec les variables
  `${alm_overall_severity}`, `${alm_n_breaches}`, `${alm_auc_native}`.
* En complément, le framework dispose d'un *dispatcher* webhook (`alerts.dispatchers`
  dans le YAML) pour notifier un canal sans passer par un reporter ; le jeton est lu dans
  une variable d'environnement, jamais dans la configuration.

### 2.8 Dashboard

Spécification complète dans `dataiku/dashboard/dashboard_monitoring.md` : une page de
synthèse métier (statut, biais de niveau, calibration, heatmap) et trois pages de
diagnostic (dérive des données, prédictions et performance, qualité). Tous les graphiques
sont des charts natifs sur `monitoring_kpis`, plus les insights du MES.

### 2.9 Plugin (industrialisation sans code)

`dataiku/plugin/alm-monitoring-plugin/` emballe la recette en **recette visuelle**
« ALM Monitoring KPIs » : l'utilisateur choisit les datasets, les colonnes (cible, score,
date, features) et les seuils dans un formulaire ; la recette construit la configuration et
exécute le monitor. C'est le mécanisme par lequel un **nouveau modèle** entre dans le
dispositif en quelques minutes.

Installation : Administration → Plugins → Development → « Add a dev plugin » → importer le
dossier (ou le dépôt Git du plugin), puis construire son code env.

## 3. Appliquer le dispositif à un nouveau modèle

1. Préparer `reference` et `current` scorés avec les colonnes du contrat.
2. Écrire `configs/<modele>.yaml` (ou remplir le formulaire du plugin) : nom, tâche,
   cible, score, date, features par type, seuils (reprendre les justifications de
   `methodologie_kpi.md` et les ancrer sur la performance hors temps du modèle).
3. `alm-monitor validate-config configs/<modele>.yaml`.
4. Dupliquer la recette Python (ou la recette du plugin) et le scénario, changer les
   variables de projet.
5. Dupliquer le dashboard et changer le filtre `model_name`.

Aucun code Python n'est modifié : c'est la définition de « holistique » retenue dans ce
projet. Le second modèle du dépôt (Bank Marketing, propension au dépôt à terme) a été
intégré exactement ainsi.

## 4. Points d'attention

* **Volumes** : `get_dataframe()` charge le dataset en mémoire ; au-delà de quelques
  millions de lignes, utiliser la recette PySpark ou limiter `lc_reference` à un
  échantillon stratifié (le PSI est stable dès 50 000 lignes).
* **Échantillonnage du MES** : la recette d'évaluation travaille sur un échantillon
  (20 000 lignes max) ; les KPIs ALM utilisent la fenêtre entière. Les deux ne sont donc
  pas strictement comparables ligne à ligne.
* **Cible différée** : en production, `lc_current` n'a la cible qu'après 12 mois ; les
  KPIs de performance renvoient `error` tant qu'elle manque (comportement voulu) et les
  indicateurs avancés portent l'alerte. Prévoir deux fenêtres : « récente sans cible »
  et « mûre avec cible ».
* **Versions** : à chaque ré-estimation, nouvelle version du Saved Model, nouvelle
  `reference` (scores hors-pli du nouveau modèle) et `model.version` incrémentée dans le
  YAML ; `monitoring_kpis` conserve `model_version` pour comparer avant/après.
* **Secrets** : clé d'API et jetons de webhook uniquement en variables d'environnement
  ou dans les *secrets* de l'utilisateur DSS.

## 5. MLflow et Dataiku : qui fait quoi

| Besoin | MLflow | Dataiku |
|---|---|---|
| Suivi d'expériences (paramètres, métriques d'entraînement) | Oui (`mlflow.db` + `mlruns/`, `scripts/export_mlflow_model.py --track`) | *Experiment tracking* natif (API MLflow intégrée) |
| Format d'échange du modèle | `mlflow.sklearn.save_model` | Import comme Saved Model MLflow |
| Monitoring en production | Non (hors périmètre de MLflow) | Model Evaluation Store + scénarios + `monitoring_kpis` |
| Registre de modèles | MLflow Model Registry | Saved Models versionnés + Deployer |

Dans ce projet, MLflow sert de **format d'échange et de trace d'entraînement** ; le
monitoring est entièrement porté par Dataiku et par ce package.

## 6. Retour d'exécution sur DSS 15.1 Free Edition

Ce que la documentation ne dit pas et qu'il a fallu découvrir en exécutant. Chaque point
est repris dans le code du dépôt.

### 6.1 Résultats obtenus

| Brique | Résultat |
|---|---|
| Recette Python `compute_monitoring_kpis` | 24 fenêtres mensuelles × 131 KPIs = **3 144 lignes**, 2 min 16 s sur 878 k lignes |
| Règle Data Quality (check Python) | **ERROR** : `CRITICAL - 33 dépassement(s) : psi[int_rate]=1.335 ; psi[application_type]=0.761 ; cramers_v[application_type]=0.347 ; js_distance[int_rate]=0.308 …` |
| Model Evaluation Store (natif, v2, 28 features) | AUC **0,642**, log-loss 0,431, calibration loss **0,026**, data drift (précision du modèle de dérive) **0,771** (p < 0,001), PSI max natif : `application_type` 0,116, `rate_spread` 0,090, `int_rate` 0,056 |
| Scénario step-based | Étape 2 (build `monitoring_kpis`) échoue d'elle-même car DSS ré-exécute la règle après build ; l'étape 5 (synthèse) tourne malgré l'échec et pose `alm_overall_severity=critical`, `alm_n_breaches=33`, `alm_auc_native=0.6417`, `alm_data_drift_native=0.7713` |
| Dashboard | 4 charts sur `monitoring_kpis` + rapport MES + bouton *Run* du scénario + statut Data Quality |

Les deux sources **concordent** (AUC 0,64 et calibration 0,026 des deux côtés) et se
complètent : le MES compare 2016-2017 en bloc à la référence (PSI `int_rate` 0,056), nos
KPIs fenêtrés isolent décembre 2017 (PSI 1,33) et datent la rupture (juin 2017). C'est
l'argument du monitoring fenêtré : une dérive récente est diluée dans une évaluation globale.

### 6.2 Écarts constatés avec la documentation, et corrections

| Constat | Correction dans le dépôt |
|---|---|
| Free Edition : pas d'API publique (pas de clé), pas de déclencheur temporel, pas de `run()` par API (`LicenseRestrictionException`) | `00_setup_project.py` bascule sur `dataiku.api_client()` quand il tourne dans DSS ; notebooks `dataiku/notebooks/` ; scénario lancé par bouton *Run* (tuile de dashboard) |
| Free Edition : pas de création de code env *managé* ; mais un modèle MLflow **refuse le builtin** (« Please specify a code environment ») | Code env `alm-monitoring` avec `mlflow`, `scikit-learn`, `cloudpickle` **et `statsmodels`** (requis par le drift natif : `No module named 'statsmodels'`) |
| Datasets uploadés : schéma 100 % `string`, *Infer types* grisé | Typage par l'API (`settings.schema_columns`) **avant** `set_core_metadata`, sinon toutes les features sont enregistrées catégorielles et le drift natif traite `loan_amnt` comme une catégorie |
| `set_core_metadata(get_features_from_dataset=…)` fige **toutes** les colonnes du dataset (dont `id`, `issue_d`, `score`) dans la signature ; le drift natif devient trivial (0,9998) et la signature ne peut plus être réduite | `features_list=[{"name", "type": "double"/"string"}]` (types de schéma DSS, pas `NUMERIC`) ; nouvelle version `v2` ; **signature MLflow explicite** dans `scripts/export_mlflow_model.py` |
| La version importée doit être **évaluée sur la référence** avant la recette d'évaluation (« The MLflow model version was not evaluated after import ») | `handler.evaluate("lc_reference")` juste après l'import |
| `ExternalModelVersionHandler` n'a pas de `set_active()` (`set_active=True` est le défaut de l'import) | Appel supprimé |
| `list_model_evaluation_stores()` renvoie des objets (`.get_settings().get_raw()["name"]`), pas des dicts | Corrigé dans le script et le notebook |
| Clé de métrique native : `dataDrift` (pas `dataDriftModelAccuracy`) | Corrigé dans `02_scenario_monitoring.py` |
| Les règles Data Quality s'exécutent **automatiquement après build** et font échouer l'étape de build du scénario | Documenté ; étapes 3-4 gardées pour la lisibilité ; étape 5 en `RUN_IF_STATUS_MATCH` SUCCESS/WARNING/FAILED |
| Versions : modèle entraîné avec scikit-learn 1.9.1, DSS embarque 1.6.1 (`InconsistentVersionWarning`, prédictions correctes) | En production, figer la **même** version des deux côtés (`requirements.txt` du modèle MLflow = code env DSS) |
| Le check Python affichait `nan` pour les KPIs de niveau modèle (`column` vide) | `isinstance(r.column, str)` |
