"""Helpers for the TimesFM-3 behaviour experiments (plan: timesfm_behaviour_plan.md): simulators, direct
predict_batch call (same flags as the official pipeline), runtime capture of network part / trend line."""
import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import timesfm_lib as tl
from tf_paths import RES_DIR

SEED_BEH = 20261125
CTX = 256
CACHE = RES_DIR / "timesfm_behaviour_cache"
CACHE.mkdir(parents=True, exist_ok=True)
LOG0 = float(np.log(100.0))


# ------------------------------------------------------------------------------------------ SPY stats
def spy_returns(a="2008-01-01", b="2018-12-31"):
    p = tl.panel()
    r = pd.Series(np.diff(p["spy"]), index=p["dates"][1:])
    return r[(r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))].to_numpy()


def fit_garch(r):
    """Gaussian MLE of r_t = mu + e_t, e_t = s_t z_t, s_t^2 = w + a e_{t-1}^2 + b s_{t-1}^2."""
    r = np.asarray(r, float)
    v0 = r.var()

    def nll(th):
        mu, lw, la, lb = th
        w, a, b = math.exp(lw), 1 / (1 + math.exp(-la)), 1 / (1 + math.exp(-lb))
        if a + b >= 0.9999:
            return 1e10
        e = r - mu
        s2 = np.empty(len(r)); s2[0] = v0
        for t in range(1, len(r)):
            s2[t] = w + a * e[t - 1] ** 2 + b * s2[t - 1]
        return 0.5 * np.sum(np.log(s2) + e ** 2 / s2)

    th0 = [r.mean(), math.log(v0 * 0.02), -2.2, 2.0]
    res = minimize(nll, th0, method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-6, fatol=1e-6))
    mu, lw, la, lb = res.x
    return dict(mu=float(mu), w=math.exp(lw), a=1 / (1 + math.exp(-la)), b=1 / (1 + math.exp(-lb)), nll=float(res.fun))


class World:
    """SPY-like constants, computed once from 2008-2018 SPY returns."""
    def __init__(self):
        self.r = spy_returns()
        self.mu, self.sigma = float(self.r.mean()), float(self.r.std(ddof=1))
        self.g = fit_garch(self.r)
        self.g["mu_fitted"] = self.g["mu"]
        self.g["mu"] = self.mu                        # drift matched to the SPY sample mean (deviation from 'fitted mu', see write-up)


def sim_returns(world: World, fam: str, n: int, L: int, rng) -> np.ndarray:
    """(n, L) daily log returns. A iid normal, B GARCH(1,1) normal, C iid bootstrap of real 2008-2018 returns."""
    if fam == "A":
        return world.mu + world.sigma * rng.standard_normal((n, L))
    if fam == "C":
        return rng.choice(world.r, size=(n, L), replace=True)
    if fam == "B":
        g, burn = world.g, 500
        z = rng.standard_normal((n, L + burn))
        s2 = np.full(n, g["w"] / (1 - g["a"] - g["b"]))
        out = np.empty((n, L + burn))
        for t in range(L + burn):
            e = np.sqrt(s2) * z[:, t]
            out[:, t] = g["mu"] + e
            s2 = g["w"] + g["a"] * e ** 2 + g["b"] * s2
        return out[:, burn:]
    raise ValueError(fam)


def to_levels(ret: np.ndarray, x0: float = LOG0) -> np.ndarray:
    """(n, L) returns -> (n, L+1) log prices starting at x0."""
    return np.concatenate([np.full((len(ret), 1), x0), x0 + np.cumsum(ret, axis=1)], axis=1)


# ------------------------------------------------------------------------------------------ model call
def fc(m, tgts, covs=None, cap=None):
    """Direct predict_batch call with exactly the official flags. tgts: list of 1-D float32; covs: list of (nc, len)."""
    import torch
    torch.manual_seed(tl.SEED)
    tgts = [np.ascontiguousarray(t, dtype=np.float32) for t in tgts]
    if covs is not None:
        covs = [np.ascontiguousarray(c, dtype=np.float32) for c in covs]
    with torch.no_grad():
        out = list(m.predict_batch(contexts=tgts, horizon=tl.H_MAX, past_only_covariates=covs,
                                   return_quantiles=True, make_positive=False,
                                   use_symmetric_averaging=False, use_znorm=False, sort_quantiles=True))
    q = np.stack([o.quantiles for o in out]).astype(np.float64)
    assert q.shape == (len(tgts), tl.H_MAX, tl.NQ) and np.isfinite(q).all()
    return q


class Capture:
    """Wrap model.forward / decode of the loaded copy (as timesfm_rootcause_capture.py) and record, per call,
    N (network quantiles, before the trend-line add-back) and Q (final quantiles), SPY/target series only."""
    def __init__(self, m):
        self.mod = m.model
        self.rec = []
        self._lg = None

    def __enter__(self):
        import torch
        from timesfm3.torch import util
        mod = self.mod
        self.f0, self.d0 = mod.forward, mod.decode

        def fwd(inputs, **kw):
            o = self.f0(inputs, **kw)
            self._lg = o["logits"]
            return o

        def dec(*a, **kw):
            res = self.d0(*a, **kw)
            lg = self._lg
            H = res.shape[2]
            ip, ext = mod.input_patch_len, mod._stitching_extract_len
            nfp = max(math.ceil((H - (ext - ip)) / ip), 1)
            n_ctx_p = lg.shape[2] - (nfp + mod.rolls - 1)
            assert nfp == 1
            idx = torch.arange(nfp) + (n_ctx_p - 1)
            hl = util.stitch_patches(lg[:, :, idx, :ext, :], ip)[:, :, :H, :]
            self.rec.append(dict(N=hl[:, 0, :tl.H_MAX, :].double().numpy(), Q=res[:, 0, :tl.H_MAX, :].double().numpy()))
            return res

        mod.forward, mod.decode = fwd, dec
        return self

    def __exit__(self, *e):
        self.mod.forward, self.mod.decode = self.f0, self.d0

    def take(self):
        r, self.rec = self.rec, []
        return {k: np.concatenate([x[k] for x in r]) for k in ("N", "Q")}


def p_up_levels(q, anchors, h):
    """Official read-out for price-level input, vectorised over rows."""
    return np.array([tl.p_up(q[i], "price", h, float(anchors[i])) for i in range(len(q))])


# ------------------------------------------------------------------------------------------ chunked cache
def run_cached(path, N, make, chunk=128, log=print, keys=("q",), cap_model=None):
    """Compute N items in chunks via make(i0, i1) -> dict of arrays (each length i1-i0); resumable npz at `path`."""
    import os, time
    st = {}
    done = 0
    if path.exists():
        z = np.load(path)
        st = {k: z[k] for k in z.files if k != "done"}
        done = int(z["done"])
    t_all = time.time()
    for i in range(done, N, chunk):
        j = min(N, i + chunk)
        t0 = time.time()
        new = make(i, j)
        for k, v in new.items():
            if k not in st:
                st[k] = np.zeros((N,) + v.shape[1:], dtype=v.dtype)
            st[k][i:j] = v
        tmp = str(path) + ".tmp.npz"
        np.savez(tmp, done=j, **st)
        os.replace(tmp, path)
        log(f"{path.name}: {j}/{N} {(time.time() - t0) / (j - i):.3f}s/fc elapsed {time.time() - t_all:.0f}s", flush=True)
    return st
