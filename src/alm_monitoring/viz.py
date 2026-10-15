"""Figures Plotly pour les notebooks et tableaux de bord (nécessite l'extra ``viz``).

Chaque fonction accepte le DataFrame au format long produit par
:meth:`MonitoringReport.to_dataframe` / :meth:`MonitoringReport.concat`, de sorte que les
mêmes graphiques peuvent être reconstruits à partir du dataset Dataiku *monitoring_kpis*.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from alm_monitoring.exceptions import ConfigurationError
from alm_monitoring.types import Severity

if TYPE_CHECKING:
    import plotly.graph_objects as go

SEVERITY_COLORS: dict[str, str] = {
    Severity.OK.value: "#2e7d32",
    Severity.WARNING.value: "#f9a825",
    Severity.CRITICAL.value: "#c62828",
    Severity.ERROR.value: "#616161",
}
_SEVERITY_LEVEL = {s.value: s.rank for s in Severity}


def _go() -> Any:
    try:
        import plotly.graph_objects as go  # noqa: PLC0415 - dépendance optionnelle
    except ImportError as exc:  # pragma: no cover
        raise ConfigurationError(
            "plotly est requis pour les graphiques ; "
            "installer avec `pip install alm-model-monitoring[viz]`"
        ) from exc
    return go


def _label(row: pd.Series[Any]) -> str:
    return f"{row['kpi']}" if pd.isna(row["column"]) else f"{row['kpi']} · {row['column']}"


def _window(df: pd.DataFrame) -> pd.Series[Any]:
    """Début de fenêtre si l'exécution a été découpée dans le temps, sinon son horodatage."""
    run_at = pd.to_datetime(df["run_at"], utc=True).dt.tz_localize(None)
    if "window_start" not in df:
        return run_at
    window = pd.to_datetime(df["window_start"], errors="coerce")
    return window.fillna(run_at)


def severity_heatmap(
    history: pd.DataFrame, *, title: str = "Statut des KPI dans le temps"
) -> go.Figure:
    """Grille fenêtre x KPI colorée par sévérité (la vue « tour de contrôle »)."""
    go = _go()
    if history.empty:
        raise ConfigurationError("L'historique est vide")
    df = history.copy()
    df["label"] = df.apply(_label, axis=1)
    df["window"] = _window(df).dt.strftime("%Y-%m")
    df["level"] = df["severity"].map(_SEVERITY_LEVEL)
    grid = df.pivot_table(index="label", columns="window", values="level", aggfunc="max")
    text = df.pivot_table(index="label", columns="window", values="value", aggfunc="mean")
    colorscale = [
        [0.0, SEVERITY_COLORS["ok"]],
        [0.33, SEVERITY_COLORS["ok"]],
        [0.34, SEVERITY_COLORS["warning"]],
        [0.66, SEVERITY_COLORS["warning"]],
        [0.67, SEVERITY_COLORS["critical"]],
        [0.9, SEVERITY_COLORS["critical"]],
        [0.91, SEVERITY_COLORS["error"]],
        [1.0, SEVERITY_COLORS["error"]],
    ]
    fig = go.Figure(
        go.Heatmap(
            z=grid.to_numpy(),
            x=grid.columns.tolist(),
            y=grid.index.tolist(),
            text=np.round(text.reindex_like(grid).to_numpy(), 3),
            texttemplate="%{text}",
            colorscale=colorscale,
            zmin=0,
            zmax=3,
            showscale=False,
            hovertemplate="%{y}<br>%{x}<br>valeur=%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title, height=max(400, 22 * len(grid.index)), margin={"l": 10, "r": 10, "t": 50}
    )
    return fig


def kpi_timeseries(
    history: pd.DataFrame, kpi: str, *, column: str | None = None, title: str | None = None
) -> go.Figure:
    """Évolution d'un KPI avec ses seuils warning / critical."""
    go = _go()
    mask = history["kpi"] == kpi
    mask &= history["column"].isna() if column is None else history["column"] == column
    df = history.loc[mask].copy()
    if df.empty:
        raise ConfigurationError(
            "Aucune ligne pour ce KPI", details={"kpi": kpi, "column": column}
        )
    df["window"] = _window(df)
    df = df.sort_values("window")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["window"],
            y=df["value"],
            mode="lines+markers",
            name=kpi,
            marker={"color": df["severity"].map(SEVERITY_COLORS), "size": 10},
            line={"color": "#37474f"},
        )
    )
    for level, dash in (("warning_threshold", "dash"), ("critical_threshold", "dot")):
        value = df[level].dropna()
        if not value.empty:
            fig.add_hline(
                y=float(value.iloc[0]),
                line={"dash": dash, "color": SEVERITY_COLORS[level.split("_")[0]]},
                annotation_text=f"seuil {level.split('_')[0]}",
            )
    fig.update_layout(
        title=title or (f"{kpi} · {column}" if column else kpi),
        xaxis_title="fenêtre",
        yaxis_title=kpi,
        height=380,
    )
    return fig


def distribution_comparison(
    reference: pd.Series[Any], current: pd.Series[Any], *, name: str, bins: int = 30
) -> go.Figure:
    """Histogrammes (numérique) ou barres (catégoriel) superposés référence vs courant."""
    go = _go()
    fig = go.Figure()
    if pd.api.types.is_numeric_dtype(reference):
        fig.add_trace(
            go.Histogram(
                x=reference, name="référence", opacity=0.6, nbinsx=bins, histnorm="probability"
            )
        )
        fig.add_trace(
            go.Histogram(
                x=current, name="courant", opacity=0.6, nbinsx=bins, histnorm="probability"
            )
        )
        fig.update_layout(barmode="overlay")
    else:
        ref_freq = reference.astype("string").value_counts(normalize=True)
        cur_freq = current.astype("string").value_counts(normalize=True)
        levels = sorted(set(ref_freq.index) | set(cur_freq.index))
        fig.add_trace(go.Bar(x=levels, y=ref_freq.reindex(levels, fill_value=0), name="référence"))
        fig.add_trace(go.Bar(x=levels, y=cur_freq.reindex(levels, fill_value=0), name="courant"))
        fig.update_layout(barmode="group")
    fig.update_layout(
        title=f"Déplacement de distribution · {name}", yaxis_title="part", height=380
    )
    return fig


def severity_summary(
    report_frame: pd.DataFrame, *, title: str = "Résultats par catégorie"
) -> go.Figure:
    """Barres empilées des sévérités par catégorie de KPI pour une seule exécution."""
    go = _go()
    counts = report_frame.pivot_table(
        index="category", columns="severity", values="kpi", aggfunc="count", fill_value=0
    )
    fig = go.Figure()
    for severity in (s.value for s in Severity):
        if severity in counts.columns:
            fig.add_trace(
                go.Bar(
                    x=counts.index,
                    y=counts[severity],
                    name=severity,
                    marker_color=SEVERITY_COLORS[severity],
                )
            )
    fig.update_layout(barmode="stack", title=title, height=360)
    return fig
