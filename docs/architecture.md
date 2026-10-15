# Architecture du framework

## 1. Principes

1. **Un contrat de données, tous les modèles.** Un modèle se déclare par un `ModelSpec`
   (nom, tâche, cible, score, date, features par type). Tout le reste en découle.
2. **Registre de KPIs ouvert.** Un KPI est une sous-classe de `KPI` décorée par
   `@register` ; le monitor, le rapport, les figures, la recette Dataiku et le plugin le
   découvrent automatiquement (`alm-monitor list-kpis`).
3. **Des rapports, pas des plantages.** Un KPI qui ne peut pas être calculé produit un
   résultat `error` ; un flux amont cassé produit un rapport rouge, pas une exception dans
   un ordonnanceur à 6 h du matin.
4. **Indicateurs avancés et retardés explicites.** Les KPIs de performance déclarent
   `requires_labels` ; sans cible, ils le disent.
5. **Sécurité par défaut.** `yaml.safe_load` uniquement, chemins confinés
   (`resolve_within`), URLs de webhook filtrées (anti-SSRF), secrets en variables
   d'environnement, téléchargements plafonnés, protection zip-slip.

## 2. Composants

```
src/alm_monitoring/
├── config.py            contrat de données et seuils (pydantic, clés inconnues refusées)
├── types.py             Severity (ordonnée), KPICategory, TaskType
├── exceptions.py        hiérarchie d'erreurs (code + détails structurés)
├── kpis/
│   ├── base.py          KPIResult, MonitoringContext, KPI (seuils, direction), registre
│   ├── _stats.py        routines numériques pures (PSI, JS, KS, Cramér's V, ECE)
│   ├── drift.py         psi, js_distance, ks_test, cramers_v, prediction_drift
│   ├── performance.py   auc, brier_score, log_loss, calibration_error, level_bias,
│   │                    performance_degradation, rmse, mae, r2
│   └── data_quality.py  schema_conformity, missing_rate, out_of_range_rate,
│                        unseen_categories_rate
├── monitor.py           ModelMonitor : run(reference, current), run_over_time(freq)
├── report.py            MonitoringReport : gravité globale, table longue, JSON
├── alerts.py            AlertManager + dispatchers logging / webhook
├── leakage.py           audit de fuite (liste noire, découpe temporelle, pouvoir
│                        discriminant unitaire, validation adversariale)
├── modeling/benchmark.py candidats, métriques, validation hors temps, sélection
├── data/
│   ├── lending_club.py  préparation du cas principal (cible à horizon, censure)
│   ├── bank_marketing.py second cas (propension au dépôt à terme)
│   ├── sources.py       CSV local, Kaggle, UCI (téléchargement plafonné)
│   └── synthetic.py     générateur pour les tests
├── spark/drift.py       backend PySpark des KPIs de dérive (mêmes formules)
├── dataiku/adapters.py  lecture/écriture de datasets DSS, check, variables de scénario
├── viz.py               figures Plotly (séries, heatmap, synthèse)
└── cli.py               alm-monitor run | list-kpis | validate-config
```

## 3. Flux d'exécution d'un run

```
MonitoringConfig (YAML) ──► ModelMonitor
                               │  valide la référence (contrat)
                               │  pour chaque KPI activé :
                               │     ctx = MonitoringContext(reference, current, model)
                               │     résultats = kpi.compute(ctx)   (erreurs → KPIResult error)
                               ▼
                         MonitoringReport
                               │  overall_severity = max(severities)
                               │  to_dataframe()  → table longue (history.csv, monitoring_kpis)
                               │  to_json()       → latest.json
                               ▼
                         AlertManager.process(report)   si gravité ≥ politique
```

`run_over_time` découpe la fenêtre courante par période (`MS` = mois) et produit un
rapport par période ; les périodes trop petites sont ignorées avec un avertissement.

## 4. Choix de conception discutables (et pourquoi)

* **Seuils absolus plutôt que statistiques.** Avec des dizaines de milliers de lignes,
  toute différence est « significative » : les p-values sont conservées en détail mais
  ne pilotent pas la gravité. Les seuils (PSI 0,10 / 0,25…) sont ceux que les validateurs
  connaissent ; ils sont surchargés par modèle.
* **Classes par quantiles de la référence.** Robuste aux queues lourdes ; une variable à
  faible cardinalité reçoit une classe par valeur pour ne pas masquer un déplacement.
* **Table longue.** Une ligne par (fenêtre, KPI, colonne) : c'est le format que les
  charts Dataiku, Plotly et un tableur exploitent sans transformation.
* **Pas de dépendance à Dataiku dans le cœur.** Le SDK est importé paresseusement ; le
  framework se teste et s'exécute partout. Les tests d'intégration Dataiku utilisent un
  faux SDK.
* **Identifiants en anglais, texte en français.** Les noms de code (`psi`, `reference`,
  `compute`) suivent la convention universelle ; docstrings, logs, documentation et
  messages sont en français, langue de l'équipe cible.

## 5. Qualité

| Garde-fou | Outil | Règle |
|---|---|---|
| Style | `ruff` (PEP 8 + 25 familles de règles dont bandit `S`) | zéro finding |
| Types | `mypy --strict` | zéro erreur |
| Tests | `pytest` (marqueurs unit / integration / security / slow) | couverture ≥ 90 % (97 % mesurés avec le backend PySpark, 93 % sans) |
| Sécurité | `bandit`, `pip-audit`, suite de tests de sécurité | zéro haut/moyen |
| Pré-commit | `pre-commit` | tout ce qui précède à chaque commit |
| CI | GitHub Actions | lint, types, tests sur Python 3.11 et 3.12 |

## 6. Évolutions prévues

* KPIs par segment avec agrégation hiérarchique des gravités.
* Référence glissante à décroissance temporelle après recalibrage validé.
* Intervalles de confiance (bootstrap) sur les KPIs pour les segments fins.
* Backend PySpark étendu aux KPIs de performance.
