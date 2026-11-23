"""ROOT CAUSE, reference: AUC of the official forecast vs the same forecast with the detrending switched off / forced on
(runs already cached by timesfm_verify_detrend.py `run`).  Writes results/timesfm_rootcause_detrend_ref.csv."""
import numpy as np
from tf_paths import RES_DIR
import timesfm_lib as tl
from timesfm_rootcause_lib import test_dates, anchor, score_from_q, evaluate, HS
from timesfm_verify_detrend import aligned2

d = test_dates(); a = anchor(d)
sc = {nm: {h: score_from_q(aligned2(k, d), a, h) for h in HS}
      for nm, k in (("official (detrend on when fit is good)", "official"), ("detrend off", "off"), ("detrend always on", "always"))}
df = evaluate(sc, d)
df.to_csv(RES_DIR / "timesfm_rootcause_detrend_ref.csv", index=False)
print(df[["model", "h", "auc_pooled", "ci_lo", "ci_hi", "d_vs_first", "d_vs_first_lo", "d_vs_first_hi"] + [f"auc_{y}" for y in range(2019, 2026)]].round(3).to_string(index=False))
