"""Check Python personnalisé sur le dataset ``monitoring_kpis``.

À coller dans Dataset → Status → Checks → « Custom Python check ». Le check lit la
dernière exécution (``run_id`` le plus récent) et renvoie :

* ``ERROR``   si au moins un KPI est ``critical`` ou ``error`` (le scénario échoue, les
  reporters configurés en « on failure » partent) ;
* ``WARNING`` si au moins un KPI est ``warning`` ;
* ``OK``      sinon.

Le message liste les cinq dépassements les plus graves pour que l'alerte soit
actionnable sans ouvrir le dashboard.
"""

import dataiku

RANK = {"ok": 0, "warning": 1, "critical": 2, "error": 3}


def process(last_values, dataset, partition_id):  # noqa: ARG001 - signature imposée par DSS
    frame = dataiku.Dataset(dataset.name).get_dataframe()
    if frame.empty:
        return "ERROR", "monitoring_kpis est vide"
    latest = frame.sort_values("run_at")["run_id"].iloc[-1]
    last = frame[frame["run_id"] == latest]
    worst = max(RANK.get(s, 0) for s in last["severity"])
    breaches = last[last["severity"] != "ok"].copy()
    breaches["rank"] = breaches["severity"].map(RANK)
    top = breaches.sort_values(["rank", "value"], ascending=[False, False]).head(5)
    detail = "; ".join(
        f"{r.kpi}[{r.column if isinstance(r.column, str) else 'modèle'}]"
        f"={r.value:.3f} ({r.severity})"
        for r in top.itertuples()
    )
    if worst >= RANK["critical"]:
        return "ERROR", f"CRITICAL - {len(breaches)} dépassement(s) : {detail}"
    if worst == RANK["warning"]:
        return "WARNING", f"WARNING - {len(breaches)} dépassement(s) : {detail}"
    return "OK", f"Aucun dépassement sur {len(last)} résultats"
