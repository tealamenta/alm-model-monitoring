"""Backend PySpark pour les KPIs de dérive et de qualité des données.

Les KPIs de dérive (PSI, distance de Jensen-Shannon) ne dépendent que de **fréquences
agrégées par classe** : Spark calcule ces agrégats de façon distribuée, puis les formules
sont appliquées côté driver avec exactement les mêmes fonctions que le backend pandas
(:mod:`alm_monitoring.kpis._stats`). Les deux backends donnent donc des valeurs
comparables, aux approximations de quantiles près (``approxQuantile``).

Ce module est conçu pour une recette **PySpark** Dataiku ou un job Spark autonome sur
l'historique complet (plusieurs millions de lignes) ; la version pandas reste la
référence pour les tests et les fenêtres mensuelles.

Usage::

    from pyspark.sql import SparkSession
    from alm_monitoring.spark.drift import compute_drift, missing_rates

    spark = SparkSession.builder.getOrCreate()
    ref = spark.read.csv("reference.csv", header=True, inferSchema=True)
    cur = spark.read.csv("current.csv", header=True, inferSchema=True)
    table = compute_drift(ref, cur, numeric=["int_rate", "fico"], categorical=["grade"])
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from alm_monitoring.exceptions import InsufficientDataError
from alm_monitoring.kpis import _stats

if TYPE_CHECKING:
    from pyspark.sql import DataFrame as SparkDataFrame

_LOG = logging.getLogger(__name__)
_DEFAULT_BINS = 10
_QUANTILE_ERROR = 0.001


def _functions() -> Any:
    from pyspark.sql import functions as f  # noqa: PLC0415 - dépendance optionnelle

    return f


def _is_missing(frame: SparkDataFrame, column: str) -> Any:
    """Condition « valeur manquante » : NULL, et NaN pour les colonnes flottantes.

    Spark distingue NULL de NaN (NaN est même supérieur à toute valeur dans les
    comparaisons) alors que pandas confond les deux ; on aligne le comportement.
    """
    f = _functions()
    condition = f.col(column).isNull()
    if dict(frame.dtypes).get(column) in {"double", "float"}:
        condition = condition | f.isnan(f.col(column))
    return condition


def reference_bin_edges(
    reference: SparkDataFrame, column: str, bins: int = _DEFAULT_BINS
) -> np.ndarray:
    """Bornes de classes par quantiles de la référence (même convention que le backend pandas)."""
    distinct = reference.select(column).where(~_is_missing(reference, column)).distinct().count()
    if distinct == 0:
        raise InsufficientDataError("Colonne vide dans la référence", details={"column": column})
    if distinct <= bins:
        values = sorted(
            float(r[0])
            for r in reference.select(column)
            .where(~_is_missing(reference, column))
            .distinct()
            .collect()
        )
        if len(values) == 1:
            # Référence constante : même règle que le backend pandas (valeur isolée dans sa
            # classe), sinon toute valeur courante tomberait dans l'unique classe et le
            # PSI vaudrait 0 malgré un changement complet.
            tol = max(abs(values[0]), 1.0) * 1e-9
            inner = np.array([values[0] - tol, values[0] + tol])
        else:
            inner = (np.asarray(values[:-1]) + np.asarray(values[1:])) / 2.0
    else:
        probabilities = list(np.linspace(0.0, 1.0, bins + 1)[1:-1])
        quantiles = reference.approxQuantile(column, probabilities, _QUANTILE_ERROR)
        inner = np.unique(np.asarray(quantiles, dtype="float64"))
    return np.concatenate(([-np.inf], inner, [np.inf]))


def binned_frequencies(
    frame: SparkDataFrame, column: str, edges: np.ndarray
) -> tuple[np.ndarray, int]:
    """Fréquences relatives par classe, calculées par un ``groupBy`` sur l'indice de classe."""
    f = _functions()
    inner = [float(e) for e in edges[1:-1]]
    # Indice de classe = nombre de bornes internes strictement inférieures à la valeur
    # (convention « right=False » de numpy.histogram sur les bornes internes).
    expr = f.lit(0)
    for edge in inner:
        expr = expr + f.when(f.col(column) >= f.lit(edge), 1).otherwise(0)
    counts = (
        frame.where(~_is_missing(frame, column))
        .select(expr.alias("bin"))
        .groupBy("bin")
        .count()
        .collect()
    )
    total = sum(int(r["count"]) for r in counts)
    if total == 0:
        raise InsufficientDataError("Aucune valeur finie à classer", details={"column": column})
    freq = np.zeros(len(edges) - 1, dtype="float64")
    for r in counts:
        freq[int(r["bin"])] = int(r["count"]) / total
    return freq, total


def categorical_frequencies(
    reference: SparkDataFrame, current: SparkDataFrame, column: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Fréquences alignées sur l'union des modalités (les manquants forment une modalité)."""
    f = _functions()

    def counts(frame: SparkDataFrame) -> dict[str, int]:
        rows = (
            frame.select(f.coalesce(f.col(column).cast("string"), f.lit("<NA>")).alias("level"))
            .groupBy("level")
            .count()
            .collect()
        )
        return {str(r["level"]): int(r["count"]) for r in rows}

    ref_counts, cur_counts = counts(reference), counts(current)
    levels = sorted(set(ref_counts) | set(cur_counts))
    n_ref, n_cur = sum(ref_counts.values()), sum(cur_counts.values())
    if n_ref == 0 or n_cur == 0:
        raise InsufficientDataError("Série catégorielle vide", details={"column": column})
    ref_freq = np.asarray([ref_counts.get(level, 0) / n_ref for level in levels])
    cur_freq = np.asarray([cur_counts.get(level, 0) / n_cur for level in levels])
    return ref_freq, cur_freq, levels


def compute_drift(
    reference: SparkDataFrame,
    current: SparkDataFrame,
    *,
    numeric: Sequence[str] = (),
    categorical: Sequence[str] = (),
    bins: int = _DEFAULT_BINS,
) -> pd.DataFrame:
    """PSI et distance de Jensen-Shannon par colonne, calculés de façon distribuée.

    Returns:
        Table pandas ``column, type, psi, js_distance, n_reference, n_current, levels``.
    """
    rows: list[dict[str, Any]] = []
    for column in numeric:
        edges = reference_bin_edges(reference, column, bins)
        ref_freq, n_ref = binned_frequencies(reference, column, edges)
        cur_freq, n_cur = binned_frequencies(current, column, edges)
        rows.append(
            {
                "column": column,
                "type": "numeric",
                "psi": _stats.psi(ref_freq, cur_freq),
                "js_distance": _stats.jensen_shannon(ref_freq, cur_freq),
                "n_reference": n_ref,
                "n_current": n_cur,
                "levels": int(ref_freq.size),
            }
        )
    for column in categorical:
        ref_freq, cur_freq, levels = categorical_frequencies(reference, current, column)
        rows.append(
            {
                "column": column,
                "type": "categorical",
                "psi": _stats.psi(ref_freq, cur_freq),
                "js_distance": _stats.jensen_shannon(ref_freq, cur_freq),
                "n_reference": int(reference.count()),
                "n_current": int(current.count()),
                "levels": len(levels),
            }
        )
    _LOG.info("Dérive Spark calculée sur %d colonnes", len(rows))
    return pd.DataFrame(rows)


def missing_rates(frame: SparkDataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Taux de valeurs manquantes par colonne en une seule passe."""
    f = _functions()
    aggregates = [f.avg(_is_missing(frame, c).cast("double")).alias(c) for c in columns]
    row = frame.select(*aggregates).collect()[0].asDict()
    return pd.DataFrame(
        {"column": list(columns), "missing_rate": [float(row[c]) for c in columns]}
    )


def drift_by_period(
    reference: SparkDataFrame,
    current: SparkDataFrame,
    *,
    timestamp: str,
    numeric: Sequence[str] = (),
    categorical: Sequence[str] = (),
    bins: int = _DEFAULT_BINS,
) -> pd.DataFrame:
    """Dérive par mois d'émission : une table par période, concaténée avec ``window_start``."""
    f = _functions()
    periods = [
        r[0]
        for r in current.select(f.date_trunc("month", f.col(timestamp)).alias("p"))
        .distinct()
        .orderBy("p")
        .collect()
    ]
    tables = []
    for period in periods:
        window = current.where(f.date_trunc("month", f.col(timestamp)) == f.lit(period))
        table = compute_drift(
            reference, window, numeric=numeric, categorical=categorical, bins=bins
        )
        table.insert(0, "window_start", pd.Timestamp(period))
        tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
