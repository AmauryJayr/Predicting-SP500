"""ROOT CAUSE, reference: how well do plain past-return scores (no model) rank SPY's next-1 / next-5 day direction in
the same test rows?  Shows whether 'following the recent move' itself loses in 2020 / 2022 / 2025.
Writes results/timesfm_rootcause_market.csv."""
import numpy as np
import pandas as pd
import timesfm_lib as tl
from tf_paths import RES_DIR
from timesfm_rootcause_lib import test_dates, evaluate, HS

d = test_dates()
p = tl.panel()
ls = pd.Series(p["spy"], index=p["dates"])
sc = {f"past_{k}d_return (momentum)": {h: ls.diff(k).reindex(d).to_numpy() for h in HS} for k in (5, 20, 60, 255)}
df = evaluate(sc, d)
df.to_csv(RES_DIR / "timesfm_rootcause_market.csv", index=False)
print(df[["model", "h", "auc_pooled", "ci_lo", "ci_hi"] + [f"auc_{y}" for y in range(2019, 2026)]].round(3).to_string(index=False))
