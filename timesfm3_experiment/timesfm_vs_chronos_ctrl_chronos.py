"""Tiny Chronos-2 runner for the controlled-swap experiment (lives in the separate `chronos2` venv).

Reads an input file written by `timesfm_vs_chronos_ctrl_run.py prep` and writes a resumable quantile cache:
    in : npz with dates (n,), x (n, T) float64 contexts, optionally cov (n, K, T), covnames, pred_len
    out: npz with dates (n_done,), q (n_done, pred_len, 21)   (written after every chunk)
The model call is the teammate's (chronos_experiment/chronos_study.py `_predict_chunk`): Chronos2Pipeline
.predict_quantiles(inputs, prediction_length, quantile_levels=21 levels), batch 64, no other options.
No project code is imported; numpy only besides chronos/torch.

    chronos2/bin/python timesfm_vs_chronos_ctrl_chronos.py IN.npz OUT.npz [--threads 8] [--limit N] [--batch 64]
"""
import argparse
import os
import time

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
import numpy as np

QLEVELS = [0.01, 0.05] + [round(0.10 + 0.05 * i, 2) for i in range(17)] + [0.95, 0.99]   # chronos_study.QLEVELS
MODEL_ID = os.environ.get("CHRONOS_MODEL", "amazon/chronos-2")   # CHRONOS_MODEL = local copy of the same weights if the HF download is unreliable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only the first N not-yet-done dates (timing / repro checks)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--save_every", type=int, default=128)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    from chronos import Chronos2Pipeline

    z = np.load(a.inp, allow_pickle=False)
    dates, x, pl = z["dates"], z["x"], int(z["pred_len"])
    cov = z["cov"] if "cov" in z.files else None
    names = [str(s) for s in z["covnames"]] if cov is not None else None
    done_d, done_q = np.zeros(0, dtype="datetime64[ns]"), np.zeros((0, pl, len(QLEVELS)))
    if os.path.exists(a.out):
        o = np.load(a.out)
        done_d, done_q = o["dates"], o["q"]
    have = set(done_d.tolist())
    todo = [i for i in range(len(dates)) if dates[i].item() not in have]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{a.inp}: {len(dates)} dates, {len(done_d)} done, {len(todo)} to do, pred_len {pl}", flush=True)
    pipe = Chronos2Pipeline.from_pretrained(MODEL_ID, device_map="cpu")
    nd, nq = [done_d], [done_q]
    t_all, n_new = time.time(), 0
    for c0 in range(0, len(todo), a.save_every):
        part = todo[c0:c0 + a.save_every]
        t0 = time.time()
        qs = []
        for b0 in range(0, len(part), a.batch):
            idx = part[b0:b0 + a.batch]
            if cov is None:
                inputs = [{"target": x[i]} for i in idx]
            else:
                inputs = [{"target": x[i], "past_covariates": {nm: cov[i, j] for j, nm in enumerate(names)}} for i in idx]
            q, _ = pipe.predict_quantiles(inputs, prediction_length=pl, quantile_levels=QLEVELS)
            for qi in q:
                arr = qi.detach().float().cpu().numpy() if hasattr(qi, "detach") else np.asarray(qi, dtype=float)
                if arr.ndim == 3:
                    arr = arr[0]                                  # variate 0 = SPY
                assert arr.shape == (pl, len(QLEVELS)), arr.shape
                qs.append(arr)
        nd.append(dates[part])
        nq.append(np.stack(qs).astype(np.float64))
        d_all, q_all = np.concatenate(nd), np.concatenate(nq)
        assert np.isfinite(q_all).all()
        o = np.argsort(d_all)
        np.savez_compressed(a.out + ".tmp.npz", dates=d_all[o], q=q_all[o])
        os.replace(a.out + ".tmp.npz", a.out)                      # atomic: the run is resumable
        n_new += len(part)
        print(f"  {n_new}/{len(todo)} new, {(time.time() - t0) / len(part):.3f}s/forecast, "
              f"elapsed {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
