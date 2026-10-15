# Audit de qualité des données - lending_club

Audit de qualité : EXPLOITABLE (WARNING)
- [ok] fingerprint: accepted_2007_to_2018Q4.csv.gz : sha256 55c16f75120f8976…
- [ok] duplicates: 0 ligne(s) identique(s), 0 identifiant(s) répété(s), 44 doublon(s) sémantique(s) sur ['loan_amnt', 'int_rate', 'term', 'issue_d', 'annual_inc', 'dti', 'fico_range_low']
- [warning] business_rules: 13772 violation(s) sur 9 règles ; pire règle : 0.32% des lignes
- [warning] missing_values: 0 colonne(s) au-delà de 20 % de manquants ; 1 manquant(s) informatif(s) lié(s) à la cible : ['mort_acc']
- [warning] outliers: 1 colonne(s) avec plus de 2 % d'observations au-delà de 5 MAD : ['revol_bal']
- [ok] distributions: 6 colonne(s) fortement asymétrique(s) (|skew| > 2) : ['annual_inc', 'dti', 'revol_bal', 'pub_rec', 'pub_rec_bankruptcies', 'inq_last_6mths'] ; sans impact sur les arbres, à transformer pour la logistique
- [ok] cardinality: 0 colonne(s) quasi constante(s) : aucune ; 0 catégorielle(s) à forte cardinalité : aucune
- [warning] multicollinearity: VIF >= 10 : ['loan_amnt', 'installment'] ; VIF entre 5 et 10 : ['term_months'] (sans impact sur les arbres, à traiter pour la logistique)
- [warning] temporal_consistency: 130 mois de 2007-06 à 2018-03 ; 0 mois vide(s), 36 mois sous 1000 lignes, 0 date(s) illisible(s)
- [ok] target_quality: taux positif 14.8% (déséquilibre modéré), de 8.0% à 17.2% selon l'année

## business_rules

| regle                                       |   violations |   part |
|:--------------------------------------------|-------------:|-------:|
| utilisation_revolving_superieure_100        |         7343 | 0.0032 |
| dti_hors_bornes [0, 100]                    |         2563 | 0.0011 |
| mensualite_incoherente_avec_annuite (>10 %) |         2106 | 0.0009 |
| revenu_annuel_nul_ou_negatif                |         1667 | 0.0007 |
| revenu_annuel_superieur_5M                  |           93 | 0.0000 |
| fico_hors_bornes [300, 850]                 |            0 | 0.0000 |
| taux_hors_bornes (0, 40]                    |            0 | 0.0000 |
| montant_nul_ou_negatif                      |            0 | 0.0000 |
| dernier_paiement_avant_emission             |            0 | 0.0000 |

## missing_values

| colonne              |   taux_manquant |   ratio_cible_manquant_vs_present |
|:---------------------|----------------:|----------------------------------:|
| emp_length_years     |          0.0610 |                            0.8600 |
| mort_acc             |          0.0267 |                            0.6934 |
| pub_rec_bankruptcies |          0.0007 |                            0.5649 |
| revol_util           |          0.0007 |                            1.0804 |
| dti                  |          0.0004 |                            1.1827 |
| inq_last_6mths       |          0.0000 |                            0.9021 |
| open_acc             |          0.0000 |                            0.9332 |
| total_acc            |          0.0000 |                            0.9332 |
| pub_rec              |          0.0000 |                            0.9332 |
| annual_inc           |          0.0000 |                            0.0000 |
| loan_amnt            |          0.0000 |                          nan      |
| verification_status  |          0.0000 |                          nan      |
| purpose              |          0.0000 |                          nan      |
| home_ownership       |          0.0000 |                          nan      |
| grade                |          0.0000 |                          nan      |
| revol_bal            |          0.0000 |                          nan      |
| term_months          |          0.0000 |                          nan      |
| fico                 |          0.0000 |                          nan      |
| installment          |          0.0000 |                          nan      |
| int_rate             |          0.0000 |                          nan      |
| application_type     |          0.0000 |                          nan      |

## outliers

| colonne              |   part_aberrante_mad5 |   part_aberrante_iqr3 |    mediane |           max |
|:---------------------|----------------------:|----------------------:|-----------:|--------------:|
| revol_bal            |                0.0273 |                0.0235 | 11453.0000 |  1698749.0000 |
| annual_inc           |                0.0175 |                0.0160 | 65000.0000 | 61000000.0000 |
| mort_acc             |                0.0061 |                0.0007 |     1.0000 |       47.0000 |
| open_acc             |                0.0044 |                0.0055 |    11.0000 |       88.0000 |
| dti                  |                0.0022 |                0.0024 |    17.8600 |      999.0000 |
| total_acc            |                0.0009 |                0.0014 |    23.0000 |      157.0000 |
| fico                 |                0.0004 |                0.0000 |   692.0000 |      847.5000 |
| installment          |                0.0000 |                0.0000 |   376.2900 |     1717.6300 |
| revol_util           |                0.0000 |                0.0000 |    52.0000 |      193.0000 |
| loan_amnt            |                0.0000 |                0.0000 | 12750.0000 |    40000.0000 |
| term_months          |                0.0000 |                0.0000 |    36.0000 |       60.0000 |
| int_rate             |                0.0000 |                0.0000 |    12.6900 |       30.9900 |
| emp_length_years     |                0.0000 |                0.0000 |     6.0000 |       10.0000 |
| pub_rec              |                0.0000 |                0.0000 |     0.0000 |       28.0000 |
| pub_rec_bankruptcies |                0.0000 |                0.0000 |     0.0000 |       11.0000 |
| inq_last_6mths       |                0.0000 |                0.0041 |     0.0000 |       28.0000 |

## distributions

| colonne              |   asymetrie |   aplatissement | transformation_conseillee                       |
|:---------------------|------------:|----------------:|:------------------------------------------------|
| annual_inc           |    305.3477 |     117594.5811 | log1p ou classes de quantiles (modèle linéaire) |
| dti                  |     29.4880 |       2011.0264 | log1p ou classes de quantiles (modèle linéaire) |
| revol_bal            |     10.1498 |        290.8568 | log1p ou classes de quantiles (modèle linéaire) |
| pub_rec              |      6.3165 |        103.9705 | log1p ou classes de quantiles (modèle linéaire) |
| pub_rec_bankruptcies |      3.4336 |         20.7170 | log1p ou classes de quantiles (modèle linéaire) |
| inq_last_6mths       |      2.0792 |          9.8321 | log1p ou classes de quantiles (modèle linéaire) |
| mort_acc             |      1.7179 |          6.9284 | aucune                                          |
| open_acc             |      1.2977 |          3.3107 | aucune                                          |
| fico                 |      1.2488 |          1.5482 | aucune                                          |
| installment          |      1.0116 |          0.7570 | aucune                                          |
| total_acc            |      0.9977 |          1.9558 | aucune                                          |
| term_months          |      0.9531 |         -1.0916 | aucune                                          |
| int_rate             |      0.7637 |          0.6597 | aucune                                          |
| loan_amnt            |      0.7632 |         -0.1331 | aucune                                          |
| emp_length_years     |     -0.1685 |         -1.5756 | aucune                                          |
| revol_util           |     -0.0522 |         -0.7929 | aucune                                          |

## cardinality

| colonne              |   modalites |   part_modalite_dominante | type        |
|:---------------------|------------:|--------------------------:|:------------|
| loan_amnt            |        1572 |                    0.0762 | numeric     |
| term_months          |           2 |                    0.7156 | numeric     |
| int_rate             |         639 |                    0.0288 | numeric     |
| installment          |       88457 |                    0.0024 | numeric     |
| annual_inc           |       80238 |                    0.0386 | numeric     |
| dti                  |        9010 |                    0.0007 | numeric     |
| fico                 |          48 |                    0.0870 | numeric     |
| emp_length_years     |          12 |                    0.3327 | numeric     |
| open_acc             |          89 |                    0.0874 | numeric     |
| total_acc            |         150 |                    0.0366 | numeric     |
| revol_bal            |       95830 |                    0.0048 | numeric     |
| revol_util           |        1421 |                    0.0050 | numeric     |
| mort_acc             |          47 |                    0.4008 | numeric     |
| pub_rec              |          44 |                    0.8344 | numeric     |
| pub_rec_bankruptcies |          13 |                    0.8793 | numeric     |
| inq_last_6mths       |          29 |                    0.5983 | numeric     |
| grade                |           7 |                    0.2961 | categorical |
| home_ownership       |           6 |                    0.4933 | categorical |
| purpose              |          14 |                    0.5730 | categorical |
| verification_status  |           3 |                    0.3905 | categorical |
| application_type     |           2 |                    0.9637 | categorical |

## multicollinearity

| colonne              |     vif |
|:---------------------|--------:|
| loan_amnt            | 53.5367 |
| installment          | 46.2968 |
| term_months          |  6.7519 |
| int_rate             |  2.6233 |
| total_acc            |  2.4858 |
| open_acc             |  2.3217 |
| pub_rec_bankruptcies |  1.8262 |
| pub_rec              |  1.7966 |
| fico                 |  1.7096 |
| revol_util           |  1.6254 |
| mort_acc             |  1.3336 |
| revol_bal            |  1.3205 |
| dti                  |  1.1842 |
| inq_last_6mths       |  1.1265 |
| emp_length_years     |  1.0470 |
| annual_inc           |  1.0126 |

## temporal_consistency

| mois    |   lignes |
|:--------|---------:|
| 2007-06 |       24 |
| 2007-07 |       63 |
| 2007-08 |       74 |
| 2007-09 |       53 |
| 2007-10 |      105 |
| 2007-11 |      112 |
| 2007-12 |      172 |
| 2008-01 |      305 |
| 2008-02 |      306 |
| 2008-03 |      402 |
| 2008-04 |      259 |
| 2008-05 |      115 |
| 2008-06 |      124 |
| 2008-07 |      141 |
| 2008-08 |      100 |
| 2008-09 |       57 |
| 2008-10 |      122 |
| 2008-11 |      209 |
| 2008-12 |      253 |
| 2009-01 |      269 |
| 2009-02 |      302 |
| 2009-03 |      324 |
| 2009-04 |      333 |
| 2009-05 |      359 |
| 2009-06 |      406 |
| 2009-07 |      411 |
| 2009-08 |      446 |
| 2009-09 |      507 |
| 2009-10 |      604 |
| 2009-11 |      662 |
| 2009-12 |      658 |
| 2010-01 |      662 |
| 2010-02 |      682 |
| 2010-03 |      828 |
| 2010-04 |      912 |
| 2010-05 |      989 |
| 2010-06 |     1105 |
| 2010-07 |     1204 |
| 2010-08 |     1175 |
| 2010-09 |     1189 |

## target_quality

|     annee |      lignes |   taux_positif |
|----------:|------------:|---------------:|
| 2007.0000 |    603.0000 |         0.0978 |
| 2008.0000 |   2393.0000 |         0.0802 |
| 2009.0000 |   5281.0000 |         0.1040 |
| 2010.0000 |  12537.0000 |         0.1059 |
| 2011.0000 |  21721.0000 |         0.1048 |
| 2012.0000 |  53367.0000 |         0.1092 |
| 2013.0000 | 134814.0000 |         0.1275 |
| 2014.0000 | 235629.0000 |         0.1405 |
| 2015.0000 | 421095.0000 |         0.1317 |
| 2016.0000 | 434407.0000 |         0.1559 |
| 2017.0000 | 443579.0000 |         0.1718 |
| 2018.0000 | 107864.0000 |         0.1570 |
