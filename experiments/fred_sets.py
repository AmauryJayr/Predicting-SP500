"""Feature sets for the FRED study, in an importable module so that parallel workers register them too."""
import pandas as pd

import fe_lib
from common import CORE_FEATURES, DATA_DIR
from fe_lib import ColumnSet, FEDataset, run, sel_cluster, sel_stable
from fred_features import load_fred

SPY4 = ["spy_ret_20d", "spy_px_vs_sma200", "spy_vol_20d", "vix_level"]


def fred_dataset(h: int) -> FEDataset:
    ds = FEDataset(h)
    fred = load_fred(pd.read_pickle(DATA_DIR / "aligned_prices.pkl")["SPY"].index).reindex(ds.index)
    assert not fred.isna().any().any()
    ds.F = pd.concat([ds.F, fred], axis=1)
    ds.fred_cols = list(fred.columns)
    ds.cols = list(ds.F.columns)
    ds.fam = {**ds.fam, "fred": ds.fred_cols}
    return ds


# feature sets: name -> (builder, twin without FRED)
fe_lib.SELECTORS.update({
    "base_fred": lambda ds, tr: ColumnSet(ds, ds.base_cols + ds.fred_cols),
    "core_fred": lambda ds, tr: ColumnSet(ds, CORE_FEATURES + ds.fred_cols),
    "spy4_fred": lambda ds, tr: ColumnSet(ds, SPY4 + ds.fred_cols),
    "spy4": lambda ds, tr: ColumnSet(ds, SPY4),
    "stable25_basefred": lambda ds, tr: ColumnSet(ds, sel_stable(ds, tr, ds.base_cols + ds.fred_cols, 25)),
    "stable25_base": lambda ds, tr: ColumnSet(ds, sel_stable(ds, tr, ds.base_cols, 25)),
    "cluster07_basefred": lambda ds, tr: ColumnSet(ds, sel_cluster(ds, tr, ds.base_cols + ds.fred_cols, 0.7)),
    "cluster07_base": lambda ds, tr: ColumnSet(ds, sel_cluster(ds, tr, ds.base_cols, 0.7)),
})
TWIN = {"base_fred": "base122", "core_fred": "core23", "spy4_fred": "spy4",
        "stable25_basefred": "stable25_base", "cluster07_basefred": "cluster07_base"}

def job(ds, sel, model, scheme, start, end):
    p, ch = run(ds, sel, model, scheme, start, end)
    return p, ch


