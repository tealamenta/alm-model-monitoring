"""Recette Python DSS : nettoyage, cible et features métier (train/serve identiques).

À coller dans une recette Python du Flow avec entrée ``lc_raw`` et sorties
``lc_reference`` (cohortes 2013-2015) et ``lc_current`` (cohortes à surveiller).

Pourquoi une recette Python et non un Prepare recipe : les mêmes fonctions
(``prepare_lending_club``, ``engineer_features``) servent à l'entraînement hors DSS et au
scoring dans DSS ; il n'y a donc **aucun écart train/serve** possible. Le Prepare recipe
reste le bon outil quand le métier doit lire et modifier les transformations lui-même :
dans ce cas, reproduire les formules de ``engineer_features`` (ratios, ``rate_spread``,
classes FICO) dans un Prepare recipe et vérifier l'égalité des sorties par un check.
"""

import dataiku

from alm_monitoring.data import lending_club as lc
from alm_monitoring.dataiku import DataikuDatasetSource

raw = DataikuDatasetSource("lc_raw").load()
prepared = lc.engineer_features(
    lc.prepare_lending_club(raw)
)  # clean=True : valeurs impossibles -> NaN
reference, current = lc.split_reference_current(prepared)

dataiku.Dataset("lc_reference").write_with_schema(reference)
dataiku.Dataset("lc_current").write_with_schema(current)
