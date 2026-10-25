"""Search space: every drift-aware configuration paired with its matched counterparts.

Each entry: dict(family, label, method=spec, primary=spec, secondary=spec|None)
  primary   = same model / features / hyperparameters, trained ONCE at the start of the
              evaluation period and never adapted (the question "does drift handling help?")
  secondary = same model with the scheduled-retraining status quo or with only the drift
              mechanism switched off (isolates the specific mechanism)
"""
LR = [{"C": 0.003}, {"C": 0.03}]
GB = {"learning_rate": 0.05, "max_depth": 2, "max_iter": 150, "min_samples_leaf": 200, "l2_regularization": 1.0}
FEATS = ["all", "core"]


def base(model, params, feats, **kw):
    d = {"kind": "wf", "model": model, "params": params, "feats": feats, "norm": None, "refit": None}
    d.update(kw)
    return d


def fixed(model, params, feats, norm=None):
    return base(model, params, feats, norm=norm)


def grid():
    E = []
    add = lambda fam, label, m, p, s=None: E.append(dict(family=fam, label=label, method=m, primary=p, secondary=s))
    models = [("LogReg", p) for p in LR]
    # 1. exponential time-decay sample weights (refit every 21 sessions, expanding history)
    for mdl, par in models:
        for f in FEATS:
            for hl in [126, 252, 504, 1008]:
                add("decay", f"{mdl}{par} {f} hl={hl}",
                    base(mdl, par, f, refit=21, half_life=hl), fixed(mdl, par, f), base(mdl, par, f, refit=21))
    for f in FEATS:
        for hl in [252, 504]:
            add("decay", f"GBDT {f} hl={hl}", base("GBDT", GB, f, refit=63, half_life=hl),
                fixed("GBDT", GB, f), base("GBDT", GB, f, refit=63))
    # 2. covariate-drift removal: trailing-window normalisation of features (model trained once)
    for mdl, par in models:
        for f in FEATS:
            for nm in ["z252", "z504", "z1260", "rank252", "rank504"]:
                add("norm", f"{mdl}{par} {f} {nm}", fixed(mdl, par, f, nm), fixed(mdl, par, f), None)
    for f in FEATS:
        for nm in ["z252", "z504", "rank252"]:
            add("norm", f"GBDT {f} {nm}", fixed("GBDT", GB, f, nm), fixed("GBDT", GB, f), None)
    # 3. prior / label-drift correction of a fixed model
    for mdl, par in models + [("GBDT", GB)]:
        for f in FEATS:
            for N in ([63, 126, 252] if mdl == "LogReg" else [126, 252]):
                for tm in ["recent", "train"]:
                    add("prior", f"{mdl}{par if mdl == 'LogReg' else ''} {f} N={N} thr={tm}",
                        {"kind": "prior", "base": fixed(mdl, par, f), "N": N, "thr_mode": tm},
                        fixed(mdl, par, f), None)
    # 4. Platt recalibration on the most recent window of the model's own OOS predictions
    for mdl, par, R, Ns in [(m, p, 21, [126, 252, 504]) for m, p in models] + [("GBDT", GB, 63, [252])]:
        for f in FEATS:
            for N in Ns:
                b = base(mdl, par, f, refit=R)
                add("platt", f"{mdl}{par if mdl == 'LogReg' else ''} {f} R={R} N={N}",
                    {"kind": "platt", "base": b, "N": N, "mode": "recent"}, fixed(mdl, par, f),
                    {"kind": "platt", "base": b, "N": N, "mode": "static"})
    # 5. Hedge-weighted ensemble of window experts (weights from delayed discounted log-loss)
    for mdl, par in models:
        for f in FEATS:
            experts = [base(mdl, par, f, refit=21, window=w) for w in [252, 504, 1008, None]]
            for eta in [20, 100, 500]:
                for lhl in [63, 252]:
                    add("hedge", f"{mdl}{par} {f} eta={eta} lhl={lhl}",
                        {"kind": "hedge", "experts": experts, "eta": eta, "loss_hl": lhl},
                        {"kind": "hedge", "experts": experts, "adaptive": False, "fixed_experts": True},
                        {"kind": "hedge", "experts": experts, "adaptive": False})
    # 6. drift-detection-triggered retraining (delayed loss stream)
    dets = [("ph", 0.005, 0.5, "logloss"), ("ph", 0.005, 2.0, "logloss"),
            ("adwin", 0.002, None, "error"), ("adwin", 0.05, None, "error")]
    for mdl, par in models:
        for f in FEATS:
            for d, delta, lam, stream in dets:
                for W in [504, None]:
                    m = {"kind": "detect", "model": mdl, "params": par, "feats": f, "norm": None,
                         "detector": d, "delta": delta, "lam": lam, "stream": stream, "window": W}
                    add("detect", f"{mdl}{par} {f} {d}(d={delta},lam={lam}) W={W}", m, fixed(mdl, par, f),
                        base(mdl, par, f, refit=63, window=W))
    # 7. online SGD logistic regression with delayed partial_fit updates
    for f in FEATS:
        for alpha in [1e-3, 1e-2]:
            for eta0 in [1e-3, 1e-2]:
                m = {"kind": "online", "feats": f, "norm": None, "alpha": alpha, "eta0": eta0}
                add("online", f"SGD {f} alpha={alpha} eta0={eta0}", m, m | {"adaptive": False}, None)
    # 8. volatility-regime-conditional models
    for mdl, par in models + [("GBDT", GB)]:
        for f in FEATS:
            for rv in ["vix_level", "vix_pct_rank_252d"]:
                if mdl == "GBDT" and rv == "vix_pct_rank_252d":
                    continue
                add("regime", f"{mdl}{par if mdl == 'LogReg' else ''} {f} {rv}",
                    {"kind": "regime", "model": mdl, "params": par, "feats": f, "norm": None, "regime_var": rv},
                    fixed(mdl, par, f), None)
    return E
