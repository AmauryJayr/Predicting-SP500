"""Recompute the numbers quoted on the slides from the saved predictions (h=5, official multi/ctx256)."""
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
R = "../results/"
P = pd.read_pickle(R + "timesfm_test_predictions.pkl")
t5 = P[("tfm.h5", "timesfm")]; t1 = P[("tfm.h1", "timesfm")]
ph = pd.read_csv(R + "timesfm_year_phases.csv")
by = pd.read_csv(R + "timesfm_test_by_year.csv"); by = by[by.role == "timesfm"]
print("year  said_up%  (report)  auc5  csv5  auc1  csv1")
rep = {2019: 40, 2020: 66, 2021: 90, 2022: 38, 2023: 29, 2024: 75, 2025: 41}
for y in range(2019, 2026):
    g5, g1 = t5[t5.index.year == y], t1[t1.index.year == y]
    c5 = by[(by.year == y) & (by.horizon == 5)].roc_auc.iloc[0]; c1 = by[(by.year == y) & (by.horizon == 1)].roc_auc.iloc[0]
    print(y, f"{g5.pred.mean()*100:.1f}", rep[y], f"{roc_auc_score(g5.y,g5.proba):.3f} {c5:.3f} {roc_auc_score(g1.y,g1.proba):.3f} {c1:.3f}",
          f"upwin {g5.y.mean()*100:.0f}%")
def seg(a, b):
    g = t5.loc[a:b]; return len(g), int(g.pred.sum()), int(g.y.sum()), g.pred.mean(), g.y.mean(), g.proba.mean()
print("2020 fall (23, 23, 7):", seg("2020-02-20", "2020-03-23"))
print("2020 rec (112, 69, 79):", seg("2020-03-24", "2020-08-31"))
reb = pd.concat([t5.loc[a:b] for a, b in [("2022-03-09","2022-03-29"),("2022-06-17","2022-08-16"),("2022-10-13","2022-11-30")]])
fal = pd.concat([t5.loc[a:b] for a, b in [("2022-01-04","2022-03-08"),("2022-03-30","2022-06-16"),("2022-08-17","2022-10-12")]])
print("2022 rebounds (90, 71%, 4%):", len(reb), reb.y.mean(), reb.pred.mean(), "meanP", reb.proba.mean(), "falls meanP", fal.proba.mean())
print("2025 rec (120, 17, 91):", seg("2025-04-09", "2025-09-30"))
print("2025 fall called up / right:", seg("2025-02-20","2025-04-08"))
