"""ARMAX baseline (+ GARCH) for SPY direction, and the ladder of dataset-specific improvements.

Model per return component c (statsmodels ARIMA = regression with ARMA(1,1) errors):
    y_s = const + beta' x_{s-1} + u_s,      u_s = phi u_{s-1} + eps_s + theta eps_{s-1}
x_{s-1} = exogenous features known at the close before day s (standardised on the training window).

Forecast made at the close of day p for days p+1 .. p+h:
    mean:      E[y_{p+k}] = const + beta' x_p + phi^(k-1) (phi u_p + theta eps_p)   (exog held at x_p for k >= 2)
    variance of a sum over steps K: sum_m a_m^2 sigma^2_{p+m},  a_m = sum_{k in K, k >= m} psi_{k-m},
               psi_0 = 1, psi_j = phi^(j-1) (phi + theta)
    sigma^2 constant (Gaussian) or GARCH(1,1)-t forecasts of eps.
P(up) = P(sum of the forecast components > 0).

Target alignment: label = 1[Close_{t+h} > Open_{t+1}]
    = 1[ intraday_{t+1} + sum_{k=2..h} (overnight_{t+k} + intraday_{t+k}) > 0 ]   (log returns)
'cc'    -> textbook: close-to-close returns summed over k = 1..h (misaligned with the label on purpose)
'split' -> intraday model over k = 1..h  +  overnight model over k = 2..h

Leakage: every quantity used at decision date d comes from closes <= d; models are fitted only on days <= d.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats
from sklearn.linear_model import LogisticRegression
from statsmodels.tsa.arima.model import ARIMA

from common import DATA_DIR
from fe_lib import FEDataset

warnings.filterwarnings("ignore")

EXOG_SETS = {
    "X0": [],
    "X1": ["vix_level", "spy_ret_20d"],                       # textbook: fear gauge + recent momentum
    "X2": ["vix_minus_rv20", "vs_vix_vix3m_ratio", "spy_ret_250d", "spy_ret_5d", "cr_hyg_minus_ief_ret20"],
    # X2 = variance risk premium, VIX term structure, 12-month time-series momentum, 1-week reversal, credit
}


class ArmaxData:
    def __init__(self, h: int):
        self.ds = ds = FEDataset(h)
        self.h = h
        spy = pd.read_pickle(DATA_DIR / "aligned_prices.pkl")["SPY"]
        cal = spy.loc[ds.index[0]:].index
        self.cal = cal
        c, o = spy["Close"], spy["Open"]
        self.comp = {"cc": np.log(c / c.shift(1)).reindex(cal).to_numpy(),
                     "intra": np.log(c / o).reindex(cal).to_numpy(),
                     "over": np.log(o / c.shift(1)).reindex(cal).to_numpy()}
        # exogenous values known at each close (ds rows end h days early -> forward-fill the tail)
        self.Xcal = ds.F.reindex(cal).ffill()
        self.pos = cal.get_indexer(ds.index)            # calendar position of each ds row
        assert (self.pos >= 0).all()


def _psi(phi, theta, n):
    return np.array([1.0] + [phi ** (j - 1) * (phi + theta) for j in range(1, n)])


class ComponentModel:
    """ARMA(1,1)-X fitted on calendar positions [a, b]; optional GARCH(1,1)-t on its innovations."""

    def __init__(self, y, X, a, b, garch):
        self.y, self.X = y, X                                   # X[s] = exog known at close s
        s = np.arange(max(a, 1), b + 1)
        Xl = X[s - 1] if X.shape[1] else None
        self.mu_x = Xl.mean(0) if Xl is not None else None
        self.sd_x = Xl.std(0) + 1e-12 if Xl is not None else None
        Z = (Xl - self.mu_x) / self.sd_x if Xl is not None else None
        res = ARIMA(y[s], exog=Z, order=(1, 0, 1), trend="c").fit()
        p = res.params
        self.c, self.phi, self.theta, self.s2 = p[0], p[-3], p[-2], p[-1]
        self.beta = np.asarray(p[1:-3]) if Z is not None else np.zeros(0)
        self.start = s[0]
        self._filter(b)
        self.garch = None
        if garch:
            e = self.eps[s[0] + 1: b + 1] * 100
            g = arch_model(e, mean="Zero", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
            om, al, be, nu = g.params["omega"] / 1e4, g.params["alpha[1]"], g.params["beta[1]"], g.params["nu"]
            self.garch = (om, al, be, nu)
            sig2 = np.full(len(y), np.nan)
            sig2[s[0]] = np.var(self.eps[s[0] + 1: b + 1])
            for t in range(s[0] + 1, len(y)):
                sig2[t] = om + al * self.eps[t - 1] ** 2 + be * sig2[t - 1]
            self.sig2_next = sig2                                # sig2[t] = variance of eps_t given closes <= t-1

    def _xz(self, t):
        return (self.X[t] - self.mu_x) / self.sd_x if len(self.beta) else np.zeros(0)

    def _filter(self, b):
        """u_t and eps_t for every calendar day >= start (uses y_t, x_{t-1}: data at close t)."""
        n = len(self.y)
        self.u = np.full(n, np.nan)
        self.eps = np.zeros(n)
        for t in range(self.start, n):
            if np.isnan(self.y[t]):
                break
            self.u[t] = self.y[t] - self.c - (self._xz(t - 1) @ self.beta if len(self.beta) else 0.0)
            if t > self.start:
                self.eps[t] = self.u[t] - self.phi * self.u[t - 1] - self.theta * self.eps[t - 1]

    def forecast(self, p, ks, h):
        """Mean and variance of sum_{k in ks} y_{p+k}, using closes <= p only."""
        base = self.c + (self._xz(p) @ self.beta if len(self.beta) else 0.0)
        u1 = self.phi * self.u[p] + self.theta * self.eps[p]
        mean = sum(base + self.phi ** (k - 1) * u1 for k in ks)
        psi = _psi(self.phi, self.theta, h)
        a = np.array([sum(psi[k - m] for k in ks if k >= m) for m in range(1, h + 1)])
        if self.garch is None:
            var_m = np.full(h, self.s2)
        else:
            om, al, be, _ = self.garch
            s1 = om + al * self.eps[p] ** 2 + be * self.sig2_next[p]
            pers = al + be
            lr = om / (1 - pers) if pers < 1 else s1
            var_m = np.array([lr + pers ** (m - 1) * (s1 - lr) for m in range(1, h + 1)])
        return mean, float(np.sum(a ** 2 * var_m))


def run(D: ArmaxData, cfg: dict, start: int, end: int, scheme: str = "fixed", window: int | None = None,
        refit_every: int = 63) -> pd.DataFrame:
    """Walk-forward P(up) for ds rows [start, end).
    cfg: target ('cc'|'split'), exog (key of EXOG_SETS), garch (bool), calibrate (bool)."""
    ds, h = D.ds, D.h
    cols = EXOG_SETS[cfg["exog"]]
    X = D.Xcal[cols].to_numpy() if cols else np.zeros((len(D.cal), 0))
    blocks = ([(start, np.arange(start, end))] if scheme == "fixed" else
              [(b, np.arange(b, min(b + refit_every, end))) for b in range(start, end, refit_every)])
    proba, thr = np.empty(end - start), np.empty(end - start)
    for t0, rows in blocks:
        b = D.pos[t0]                                            # last calendar day whose data is known
        a = 0 if window is None else max(0, b - window + 1)
        if cfg["target"] == "cc":
            parts = [(ComponentModel(D.comp["cc"], X, a, b, cfg["garch"]), list(range(1, h + 1)))]
        else:
            parts = [(ComponentModel(D.comp["intra"], X, a, b, cfg["garch"]), list(range(1, h + 1)))]
            if h > 1:
                parts.append((ComponentModel(D.comp["over"], X, a, b, cfg["garch"]), list(range(2, h + 1))))
        nu = min([m.garch[3] for m, _ in parts if m.garch is not None], default=None)

        def zscore(p):
            mv = [m.forecast(p, ks, h) for m, ks in parts]
            return sum(x[0] for x in mv) / np.sqrt(sum(x[1] for x in mv))

        known = np.arange(ds.n_known(t0))                        # rows whose label is known at t0
        if cfg.get("calibrate"):
            # map the forecast z = E[return] / sd to P(up) with a logistic fit on the training rows:
            # returns are negatively skewed, so P(r > 0) is above Phi(z) and the textbook mapping is miscalibrated
            tr = known[D.pos[known] > max(a, 1) + 1]
            zt = np.array([zscore(D.pos[i]) for i in tr])
            cal = LogisticRegression(C=1e6).fit(zt[:, None], ds.y[tr])
        for r in rows:
            z = zscore(D.pos[r])
            if cfg.get("calibrate"):
                proba[r - start] = cal.predict_proba([[z]])[0, 1]
            else:
                proba[r - start] = stats.t.cdf(z * np.sqrt(nu / (nu - 2)), nu) if nu else stats.norm.cdf(z)
        thr[rows - start] = ds.y[known].mean()                   # same rule as every other model
    pred = pd.DataFrame({"y": ds.y[start:end], "proba": proba, "thr": thr}, index=ds.index[start:end])
    pred["pred"] = (pred["proba"] >= pred["thr"]).astype(int)
    return pred


def always_up(D: ArmaxData, start: int, end: int) -> pd.DataFrame:
    ds = D.ds
    t = ds.y[: ds.n_known(start)].mean()
    pred = pd.DataFrame({"y": ds.y[start:end], "proba": t, "thr": t, "pred": 1}, index=ds.index[start:end])
    return pred
