"""Figure statique de synthèse pour le README (matplotlib, à partir de history.csv).

Trois panneaux sur 24 mois : biais de niveau signé, AUC et erreur de calibration, nombre
de dépassements par gravité. Les figures interactives Plotly restent dans
``reports/<modele>/figures/`` ; celle-ci sert uniquement à l'aperçu GitHub.

Usage::

    python scripts/make_readme_figure.py [--history reports/lending_club/history.csv]
                                         [--out reports/lending_club/figures/synthese.png]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLORS = {"warning": "#E69F00", "critical": "#D55E00", "ok": "#009E73", "line": "#0072B2"}


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--history", type=Path, default=Path("reports/lending_club/history.csv"))
    parser.add_argument(
        "--out", type=Path, default=Path("reports/lending_club/figures/synthese.png")
    )
    args = parser.parse_args(argv)
    history = pd.read_csv(args.history)
    history["mois"] = pd.to_datetime(history["window_start"])

    def series(kpi: str, field: str = "value") -> pd.Series:
        sub = history.loc[history["kpi"] == kpi].set_index("mois").sort_index()
        return sub[field].astype(float)

    bias = series("level_bias", "detail_signed_bias") * 100
    auc = series("auc")
    ece = series("calibration_error")
    counts = history.pivot_table(
        index="mois", columns="severity", values="kpi", aggfunc="size", fill_value=0
    ).sort_index()

    fig, axes = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True)
    ax = axes[0]
    ax.plot(bias.index, bias, color=COLORS["line"], lw=2)
    ax.axhline(0, color="black", lw=0.8)
    for level, color in ((-2, COLORS["warning"]), (-4, COLORS["critical"])):
        ax.axhline(level, color=color, ls="--", lw=1)
    ax.set_ylabel("Biais de niveau (points)")
    ax.set_title(
        "Modèle de remboursement anticipé : le classement tient, le niveau dérive", loc="left"
    )
    ax.annotate(
        "prédit < observé : écoulement anticipé sous-estimé",
        xy=(bias.index[-1], bias.iloc[-1]),
        xytext=(-330, 22),
        textcoords="offset points",
        fontsize=9,
    )

    ax = axes[1]
    ax.plot(auc.index, auc, color=COLORS["ok"], lw=2, label="AUC (classement)")
    ax.set_ylim(0.55, 0.70)
    ax.set_ylabel("AUC")
    ax2 = ax.twinx()
    ax2.plot(ece.index, ece, color=COLORS["critical"], lw=2, label="Erreur de calibration")
    ax2.axhline(0.02, color=COLORS["warning"], ls="--", lw=1)
    ax2.axhline(0.04, color=COLORS["critical"], ls="--", lw=1)
    ax2.set_ylim(0, 0.06)
    ax2.set_ylabel("ECE")
    lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(lines, labels, loc="lower left", fontsize=9, ncol=2)

    ax = axes[2]
    bottom = pd.Series(0.0, index=counts.index)
    for severity in ("warning", "critical"):
        if severity in counts:
            ax.bar(
                counts.index,
                counts[severity],
                bottom=bottom,
                width=20,
                color=COLORS[severity],
                label=severity,
            )
            bottom = bottom + counts[severity]
    ax.set_ylabel("Dépassements / mois")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlabel("Cohorte d'émission (fenêtre mensuelle), référence 2013-2015")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
