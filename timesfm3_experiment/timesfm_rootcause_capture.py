"""ROOT CAUSE, experiment A (step 1 of 2): re-run the OFFICIAL TimesFM-3 forecast (multi, ctx 256, default
settings) for every 2019-2025 test day and capture, at runtime (wrapping model.forward / model.decode of our own
loaded copy; the installed package is not touched), the pieces the official forecast is made of:
    Q   final quantiles (n, 5, 9)                         = what the official pipeline returns
    N   the network's own quantiles, SPY series (n, 5, 9)  = logits after reverse RevIN, BEFORE the line add-back
    RM, RS  RevIN running mean / std used for the output patch (n,)   (SPY series)
    nfp number of forecast patches used by the stitching (asserted == 1, so RM/RS are the un-refined context stats)
Q - N is the detrending line's add-back (0 on days where detrending is not applied).
Resumable; writes results/timesfm_rootcause_cache/capture.npz (git-ignored).  Run:  python timesfm_rootcause_capture.py
"""
import math
import sys
import time

import numpy as np
import torch

import timesfm_lib as tl
from tf_paths import RES_DIR
from timesfm_verify_detrend import test_dates, get_model, set_mode, official_q

CACHE = RES_DIR / "timesfm_rootcause_cache"
CACHE.mkdir(parents=True, exist_ok=True)
FILE = CACHE / "capture.npz"
CTX, H5 = 256, 5


def load():
    if not FILE.exists():
        return None
    z = np.load(FILE)
    return {k: z[k] for k in z.files}


def main(chunk=64):
    from timesfm3.torch import util
    dates = test_dates()
    st = load()
    have = set(st["dates"].astype("datetime64[ns]").tolist()) if st else set()
    todo = [d for d in dates if d.to_datetime64().astype("datetime64[ns]").tolist() not in have]
    print(f"{len(have)} cached, {len(todo)} to do", flush=True)
    m = get_model()
    set_mode(m, "default")
    mod = m.model
    assert mod.use_linear_detrending and mod.linear_detrending_threshold == 0.5 and mod.use_stitching
    cap = {}
    f0, d0 = mod.forward, mod.decode

    def fwd(inputs, **kw):
        o = f0(inputs, **kw)
        cap["lg"], cap["stats"] = o["logits"], o["revin_stats"]
        return o

    def dec(*a, **kw):
        cap["lg"] = None
        res = d0(*a, **kw)
        lg = cap["lg"]
        H = res.shape[2]
        ip, ext = mod.input_patch_len, mod._stitching_extract_len
        nfp = max(math.ceil((H - (ext - ip)) / ip), 1)
        n_ctx_p = lg.shape[2] - (nfp + mod.rolls - 1)
        idx = torch.arange(nfp) + (n_ctx_p - 1)
        hl = util.stitch_patches(lg[:, :, idx, :ext, :], ip)[:, :, :H, :]
        mu, sd = cap["stats"]
        cap["rec"].append(dict(N=hl[:, 0, :H5, :].double().numpy(), Q=res[:, 0, :H5, :].double().numpy(),
                               RM=mu[:, 0, n_ctx_p - 1].double().numpy(), RS=sd[:, 0, n_ctx_p - 1].double().numpy(),
                               nfp=nfp, n_ctx_p=n_ctx_p))
        return res

    mod.forward, mod.decode = fwd, dec
    t_all = time.time()
    for i in range(0, len(todo), chunk):
        part = todo[i:i + chunk]
        cap["rec"] = []
        t0 = time.time()
        with torch.no_grad():
            q = tl.forecast(m, "multi", CTX, part)
        r = cap["rec"]
        assert all(x["nfp"] == 1 and x["n_ctx_p"] * mod.input_patch_len == CTX for x in r)
        new = {k: np.concatenate([x[k] for x in r]) for k in ("N", "Q", "RM", "RS")}
        assert len(new["N"]) == len(part)
        # the captured Q must equal the pipeline output (sort_quantiles=True may reorder: compare sorted)
        d = np.abs(np.sort(new["Q"], -1) - q).max()
        dof = np.abs(q - official_q(part)).max()
        new["dates"] = np.array([p.to_datetime64() for p in part])
        new["Qout"] = q
        if st is None:
            st = new
        else:
            st = {k: np.concatenate([st[k], new[k]]) for k in new}
        o = np.argsort(st["dates"])
        np.savez_compressed(FILE, **{k: v[o] for k, v in st.items()})
        st = {k: v[o] for k, v in st.items()}
        print(f"{len(st['dates'])}/{len(dates)}  {(time.time() - t0) / len(part):.2f}s/fc  "
              f"|Qcap-Qout|={d:.1e} |Qout-official cache|={dof:.1e}  elapsed {(time.time() - t_all) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
