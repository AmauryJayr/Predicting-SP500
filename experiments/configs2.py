"""Round-2 search space (designed after inspecting round-1 VALIDATION results only; no test data seen)."""
from configs import GB, LR, base, fixed


def grid():
    E = []
    add = lambda fam, label, m, p, s=None: E.append(dict(family=fam, label=label, method=m, primary=p, secondary=s))
    models = [("LogReg", p) for p in LR]
    # covariate normalisation + time-decay refits (both mechanisms); secondary isolates decay
    for mdl, par in models:
        for f in ["all", "core"]:
            for nm in ["rank252", "z1260"]:
                for hl in [252, 504]:
                    add("norm+decay", f"{mdl}{par} {f} {nm} hl={hl}",
                        base(mdl, par, f, norm=nm, refit=21, half_life=hl), fixed(mdl, par, f),
                        fixed(mdl, par, f, nm))
    for f in ["all", "core"]:
        for hl in [126, 1008]:
            add("decay", f"GBDT {f} hl={hl}", base("GBDT", GB, f, refit=63, half_life=hl),
                fixed("GBDT", GB, f), base("GBDT", GB, f, refit=63))
        add("norm+decay", f"GBDT {f} rank252 hl=252", base("GBDT", GB, f, norm="rank252", refit=63, half_life=252),
            fixed("GBDT", GB, f), fixed("GBDT", GB, f, "rank252"))
    # regime-conditional on rank-normalised features
    for mdl, par in models:
        for f in ["all", "core"]:
            add("regime", f"{mdl}{par} {f} rank252 vix_pct_rank_252d",
                {"kind": "regime", "model": mdl, "params": par, "feats": f, "norm": "rank252",
                 "regime_var": "vix_pct_rank_252d"}, fixed(mdl, par, f), fixed(mdl, par, f, "rank252"))
    # diverse-expert hedge
    lr = LR[0]
    experts = [base("LogReg", lr, "core", refit=21), base("LogReg", lr, "core", refit=21, window=252),
               base("LogReg", lr, "core", norm="rank252", refit=21), base("LogReg", lr, "all", refit=21),
               base("GBDT", GB, "core", refit=63, half_life=252), base("GBDT", GB, "all", refit=63)]
    for eta in [20, 100, 500]:
        for lhl in [63, 252]:
            add("hedge", f"diverse eta={eta} lhl={lhl}",
                {"kind": "hedge", "experts": experts, "eta": eta, "loss_hl": lhl},
                {"kind": "hedge", "experts": experts, "adaptive": False, "fixed_experts": True},
                {"kind": "hedge", "experts": experts, "adaptive": False})
    # Platt-recent on slower-refit / GBDT bases
    for mdl, par, f, R in [("LogReg", lr, "core", 63), ("LogReg", lr, "all", 63), ("GBDT", GB, "all", 63),
                           ("GBDT", GB, "core", 63)]:
        for N in [252, 504]:
            b = base(mdl, par, f, refit=R)
            add("platt", f"{mdl}{par if mdl == 'LogReg' else ''} {f} R={R} N={N}",
                {"kind": "platt", "base": b, "N": N, "mode": "recent"}, fixed(mdl, par, f),
                {"kind": "platt", "base": b, "N": N, "mode": "static"})
    return E
