# Captures d'écran — Dataiku DSS 15.1 Free Edition (15 octobre 2026)

| Fichier | Écran | Ce qu'on y lit |
|---|---|---|
| `00_projet.png` | Accueil du projet | Timeline de la mise en place, statut qualité global **Error**, 1 scénario, 2 notebooks |
| `01_flow.png` | Flow | `lc_reference` + `lc_current` → recette Python → `monitoring_kpis` ; Saved Model MLflow + `lc_current` → recette d'évaluation → Model Evaluation Store |
| `02_dashboard_haut.png` | Dashboard (haut) | Biais de niveau signé, erreur de calibration (seuil 0,025), PSI par feature (seuils 0,10 / 0,25) |
| `03_dashboard_bas.png` | Dashboard (bas) | Dépassements par mois et gravité, bouton *Run* du scénario, statut Data Quality, rapport MES natif (AUC 0,642, calibration 0,026, data drift 0,771) |
| `04_data_quality.png` | `monitoring_kpis` → Data Quality | Règle Python en **Error** : `CRITICAL - 33 dépassement(s) : psi[int_rate]=1.335 …`, ré-exécutée automatiquement après build |
| `05_scenario_runs.png` | Scénario → Last runs | Étape 1 OK, étape 2 en échec **par la règle de qualité** (voulu), étape 5 exécutée malgré l'échec |
| `06_mes_evaluations.png` | Model Evaluation Store | Historique des évaluations : v1 (signature polluée, drift 1,000) puis v2 (28 features, drift 0,771) — la correction de signature MLflow est visible |
| `07_chart_biais_de_niveau.png` | Chart | \|biais\| et biais signé par mois, seuils −0,02 / −0,04 |
| `08_chart_calibration.png` | Chart | ECE par mois, seuil 0,025 franchi en juillet 2016 |
| `09_chart_psi_features.png` | Chart | PSI par feature : `rate_spread` (pic fin 2016), `int_rate` (rupture juin 2017), `application_type` (dérive progressive) |
| `10_chart_depassements.png` | Chart | Warnings et critiques par mois : 7 → 33 |
| `11_scenario.png` | Liste des scénarios | Scénario step-based, déclencheurs non disponibles en Free Edition |
