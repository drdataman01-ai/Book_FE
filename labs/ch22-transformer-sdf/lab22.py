"""Lab 22: Does Attention Across Assets Price Returns Better?

Chapter 22 of *Financial Engineering and the Evidence: Active Learning by Verifying Famous Claims*.

CLAIM ON TRIAL
  "Attention across assets improves an SDF's out-of-sample pricing and
  Sharpe ratio."

WHAT THE LAB DOES
  The original study (Kelly, Kuznetsov, Malamud and Xu 2025) uses stock-level
  characteristics, which need paid CRSP and Compustat data. This lab
  replicates the mechanism one level up: its assets are the 153 U.S. factors of
  Jensen, Kelly and Pedersen (JKP), each described every month by five numbers
  known at the end of the month (past 1-, 3- and 12-month returns, 12-month
  volatility, 36-month average return). Every model is re-estimated each
  month on the previous 60 months and judged on the following month.

  Part A  Simple baselines: the SDF from the Fama-French five factors plus
          momentum, and the shrinkage SDF of Kozak, Nagel and Santosh (2020)
          on all 153 factors, with the penalty chosen by cross-validation.
  Part B  No attention: each factor's weight depends only on its own
          characteristics, linearly and through a shallow random-feature
          network.
  Part C  Linear attention: w_t = (X_t W X_t') X_t lambda, estimated as a
          closed-form ridge regression on the 125 products of three
          characteristics.
  Part D  A small nonlinear transformer (one to four blocks), trained to
          maximize the SDF's Sharpe ratio and averaged over random starts.
          Needs PyTorch; skipped with a message if it is not installed.
  Part E  Scoring: out-of-sample Sharpe ratios, alphas of every model against
          every other, and pricing errors of the 153 factors, for the full
          period and after 2002.
  Robustness: a placebo with shuffled characteristics, trading costs, a
  120-month window, and a comparison with a simple factor-timing rule.

DATA (all free)
  JKP factor returns, U.S., monthly (jkpfactors.com), and their theme
  clusters; Kenneth R. French Data Library: Fama/French 5 Factors (2x3) and
  the Momentum Factor.

USAGE
  python lab22.py --jkp-file data/jkp_usa_factors.csv --clusters-file data/cluster_labels.csv
  python lab22.py --source sim            # fake data, to test the code

OUTPUT
  report/lab22_report.md, three figures, CSV tables and summary.json.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from fintest import plots
from fintest.data import DataUnavailable, load_french_factors
from fintest.sdf import (
    CHARACTERISTICS,
    asset_weights,
    attention_scores,
    characteristic_array,
    features_linear,
    features_linear_attention,
    managed_returns,
    random_feature_map,
    read_jkp_clusters,
    read_jkp_factors,
    walk_forward_sdf,
)
from fintest.stats import ols_newey_west

ROOT = Path(__file__).resolve().parent
SPLIT = "2002-01"
JKP_PAGE = "https://jkpfactors.com"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _find(path, words):
    if path:
        return Path(path)
    d = ROOT / "data"
    for f in sorted(d.glob("*.csv")) if d.exists() else []:
        if all(w in f.name.lower() for w in words):
            return f
    return None


def load(source: str, files: dict, sim_seed: int = 0) -> dict:
    if source == "sim":
        return simulated(sim_seed)
    jf = _find(files.get("jkp"), ["jkp"]) or _find(files.get("jkp"), ["factor"])
    if jf is None:
        raise DataUnavailable(
            "The JKP factor returns were not found.\n"
            f"Download the U.S. monthly factor returns (all 153 factors) from {JKP_PAGE}, save the CSV in data/ with "
            "'jkp' in its name, or pass --jkp-file PATH. The theme clusters are on the same site; pass them with "
            "--clusters-file.\nTo test the code without them, use --source sim.")
    R = read_jkp_factors(jf, files.get("weighting"))
    cf = _find(files.get("clusters"), ["cluster"])
    clusters = read_jkp_clusters(cf).reindex(R.columns).fillna("unassigned") if cf else \
        pd.Series("unassigned", index=R.columns)
    ff5 = load_french_factors("ff5_monthly", path=files.get("ff5"))
    mom = load_french_factors("momentum_monthly", path=files.get("mom"))
    ff6 = ff5[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]].join(mom.iloc[:, :1].rename(columns=lambda c: "Mom"), how="inner")
    return {"R": R, "clusters": clusters, "ff6": ff6, "simulated": False, "source_file": str(jf)}


def simulated(seed: int = 0, n_clusters: int = 13, per: int = 12, n_months: int = 744, start: str = "1963-01") -> dict:
    """Fake factor returns in theme clusters. Expected returns depend on each factor's own past returns and, more
    strongly, on the past returns of its cluster peers, so attention has something to find. TESTING ONLY."""
    rng = np.random.default_rng(seed)
    N = n_clusters * per - 3  # 153
    clus = np.repeat(np.arange(n_clusters), per)[:N]
    idx = pd.period_range(start, periods=n_months, freq="M", name="date")
    R = np.zeros((n_months, N))
    mkt = 0.005 + 0.045 * rng.standard_normal(n_months)
    beta = rng.normal(0, 0.15, N)
    cshock = 0.012 * rng.standard_normal((n_months, n_clusters))
    for t in range(n_months):
        if t >= 12:
            past = R[t - 12:t].sum(axis=0)
            peer = np.array([past[clus == c].mean() for c in range(n_clusters)])[clus]
            mu = 0.002 + 0.02 * (peer - peer.mean()) + 0.005 * (past - past.mean())
        else:
            mu = np.full(N, 0.002)
        R[t] = mu + beta * (mkt[t] - 0.005) + cshock[t, clus] + 0.02 * rng.standard_normal(N)
    names = [f"f{i:03d}" for i in range(N)]
    themes = ["value", "momentum", "quality", "investment", "profitability", "low_risk", "size", "accruals",
              "debt_issuance", "seasonality", "skewness", "short_term_reversal", "profit_growth"]
    ff6 = pd.DataFrame({"Mkt-RF": mkt, "SMB": 0.002 + 0.03 * rng.standard_normal(n_months),
                        "HML": 0.003 + 0.03 * rng.standard_normal(n_months),
                        "RMW": 0.003 + 0.02 * rng.standard_normal(n_months),
                        "CMA": 0.003 + 0.02 * rng.standard_normal(n_months),
                        "Mom": 0.006 + 0.04 * rng.standard_normal(n_months)}, index=idx)
    return {"R": pd.DataFrame(R, index=idx, columns=names),
            "clusters": pd.Series([themes[c] for c in clus], index=names), "ff6": ff6, "simulated": True,
            "source_file": "simulated"}


def prepare(d: dict) -> dict:
    """Align months, keep factors with enough history, and build the characteristic array."""
    R = d["R"].copy()
    R = R.loc[:, R.notna().sum() >= 120]
    X = characteristic_array(R)
    Rnext = R.shift(-1).to_numpy()
    return {"R": R, "X": X, "Rnext": Rnext, "dates": R.index, "clusters": d["clusters"].reindex(R.columns)}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def run_models(p: dict, ff6: pd.DataFrame, window: int = 60, n_rf: int = 64, seed: int = 0,
               placebo: bool = False) -> dict:
    X, Rn, dates = p["X"], p["Rnext"], p["dates"]
    if placebo:  # shuffle characteristics across assets each month
        rng = np.random.default_rng(seed + 99)
        X = np.stack([X[t][rng.permutation(X.shape[1])] for t in range(X.shape[0])])
    out = {}
    # Part A: Fama-French six factors (weights on the six factors themselves)
    F6 = ff6.reindex(dates).shift(-1).to_numpy()
    out["FF5 + momentum"] = walk_forward_sdf(F6, window)
    # Part A: KNS shrinkage on all factors (identity map)
    out["KNS shrinkage"] = walk_forward_sdf(np.nan_to_num(Rn), window)
    out["KNS shrinkage"]["map"] = None
    # Part B: no attention, linear and random features
    rf = random_feature_map(n_rf, len(CHARACTERISTICS), seed=seed)
    for name, fmap in (("No attention, linear", features_linear), ("No attention, random features", rf),
                       ("Linear attention", features_linear_attention)):
        F = managed_returns(X, Rn, fmap)
        res = walk_forward_sdf(F, window)
        res["map"] = fmap
        res["F"] = F
        out[name] = res
    for k, v in out.items():
        v["ret"] = pd.Series(v["ret"], index=dates)  # return earned in month t+1, stored at t
    out["_X"] = X
    return out


def timing_rule(p: dict) -> pd.Series:
    """A simple factor-timing rule: overweight factors with high past-12-month returns and low volatility."""
    X, Rn = p["X"], np.nan_to_num(p["Rnext"])
    k1, k2 = CHARACTERISTICS.index("ret_12"), CHARACTERISTICS.index("vol_12")
    w = X[:, :, k1] - X[:, :, k2]
    return pd.Series((w * Rn).sum(axis=1) / X.shape[1], index=p["dates"])


def transformer_sdf(p: dict, blocks: int = 2, window: int = 60, refit: int = 12, seeds: int = 10, d_model: int = 16,
                    epochs: int = 200, lr: float = 3e-3, l2: float = 1e-4) -> pd.Series | None:
    """A small transformer across assets (softmax attention, residuals, layer norm), trained on the previous
    `window` months to maximize the Sharpe ratio of the SDF portfolio, refitted every `refit` months and averaged
    over `seeds` random starts. Returns None if PyTorch is not installed."""
    try:
        import torch
        from torch import nn
    except ImportError:
        return None

    class Block(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.att = nn.MultiheadAttention(d, num_heads=2, batch_first=True)
            self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

        def forward(self, h):
            a, _ = self.att(h, h, h, need_weights=False)
            h = self.n1(h + a)
            return self.n2(h + self.ff(h))

    class Net(nn.Module):
        def __init__(self, k, d, L):
            super().__init__()
            self.inp = nn.Linear(k, d)
            self.blocks = nn.ModuleList([Block(d) for _ in range(L)])
            self.out = nn.Linear(d, 1)

        def forward(self, x):  # x: (months, assets, k)
            h = self.inp(x)
            for b in self.blocks:
                h = b(h)
            return self.out(h).squeeze(-1)  # weights (months, assets)

    X = torch.tensor(p["X"], dtype=torch.float32)
    Rn = torch.tensor(np.nan_to_num(p["Rnext"]), dtype=torch.float32)
    T = X.shape[0]
    ret = np.full(T, np.nan)
    nets = []
    for t in range(window, T):
        if (t - window) % refit == 0:
            nets = []
            for s in range(seeds):
                torch.manual_seed(s)
                net = Net(X.shape[2], d_model, blocks)
                opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=l2)
                xw, rw = X[t - window:t], Rn[t - window:t]  # returns of t-window+1 .. t, all known at t
                for _ in range(epochs):
                    opt.zero_grad()
                    port = (net(xw) * rw).sum(dim=1) / X.shape[1]
                    loss = -port.mean() / (port.std() + 1e-8)
                    loss.backward()
                    opt.step()
                nets.append(net.eval())
        with torch.no_grad():
            w = torch.stack([n(X[t:t + 1])[0] for n in nets]).mean(dim=0)
            w = w / (w.abs().sum() + 1e-12)
            ret[t] = float((w * Rn[t]).sum())
    return pd.Series(ret, index=p["dates"])


# ---------------------------------------------------------------------------
# Part E: scoring
# ---------------------------------------------------------------------------


def _sharpe(x: pd.Series) -> float:
    x = x.dropna()
    return float(math.sqrt(12) * x.mean() / x.std(ddof=1)) if len(x) > 12 and x.std() > 0 else float("nan")


def pricing_error(R: pd.DataFrame, sdf: pd.Series) -> float:
    """Root-mean-square alpha (annual) of the assets' next-month returns on the SDF portfolio's return."""
    y = R.shift(-1).reindex(sdf.index)
    ok = sdf.notna()
    x = sdf[ok].to_numpy()
    Y = y[ok]
    Y = Y.loc[:, Y.notna().mean() > 0.9].fillna(0).to_numpy()
    reg = ols_newey_west(Y, x, lags=0)
    return float(12 * np.sqrt(np.mean(reg["coef"][0] ** 2)))


def score(models: dict, R: pd.DataFrame, split: str = SPLIT, extra: dict | None = None) -> dict:
    rets = {k: v["ret"] for k, v in models.items() if not k.startswith("_")}
    if extra:
        rets.update({k: v for k, v in extra.items() if v is not None})
    df = pd.DataFrame(rets).dropna()
    s = pd.Period(split, "M")
    rows = []
    for k in df.columns:
        rows.append({"model": k, "sharpe": _sharpe(df[k]), f"sharpe_from_{split[:4]}": _sharpe(df[k][df.index >= s]),
                     "pricing_error": pricing_error(R, df[k]),
                     f"pricing_error_from_{split[:4]}": pricing_error(R, df[k][df.index >= s]),
                     "months": len(df)})
    table = pd.DataFrame(rows)
    alphas = pd.DataFrame(index=df.columns, columns=df.columns, dtype=float)
    tstats = alphas.copy()
    for a in df.columns:
        for b in df.columns:
            if a == b:
                continue
            vol = 0.10 / math.sqrt(12)  # both portfolios scaled to 10% annual volatility
            ya = df[a] / df[a].std() * vol
            xb = df[b] / df[b].std() * vol
            reg = ols_newey_west(ya.to_numpy()[:, None], xb.to_numpy(), lags=3)
            alphas.loc[a, b] = 12 * float(reg["coef"][0, 0])
            tstats.loc[a, b] = float(reg["t"][0, 0])
    return {"table": table, "alpha": alphas, "alpha_t": tstats, "returns": df}


def costs(models: dict, p: dict, cost: float) -> pd.DataFrame:
    """Net Sharpe ratios when each unit of change in the weights on the factors costs `cost`. The factors are
    themselves long-short portfolios, so this understates the true cost."""
    rows = []
    X = models["_X"]
    for k, v in models.items():
        if k.startswith("_") or k == "FF5 + momentum":
            continue
        th = v["theta"]
        if v.get("map") is None:  # KNS: weights are the coefficients themselves
            W = th
        else:
            W = asset_weights(X, th, v["map"])
        dW = np.r_[np.nan, np.nansum(np.abs(np.diff(W, axis=0)), axis=1)]
        r = v["ret"]
        net = r - cost * pd.Series(dW, index=r.index)
        rows.append({"model": k, "avg_weight_change": float(np.nanmean(dW[np.isfinite(r.to_numpy())])),
                     "sharpe_gross": _sharpe(r), "sharpe_net": _sharpe(net)})
    return pd.DataFrame(rows)


def attention_by_cluster(models: dict, p: dict, every: int = 12) -> pd.DataFrame:
    """Average attention score between theme clusters (rows: whose weight; columns: whose characteristics),
    averaged over every `every`-th month for speed."""
    v = models["Linear attention"]
    X = models["_X"]
    cl = p["clusters"].to_numpy()
    names = sorted(set(cl))
    acc = np.zeros((len(names), len(names)))
    n = 0
    pos = {c: i for i, c in enumerate(names)}
    ids = np.array([pos[c] for c in cl])
    for t in range(0, X.shape[0], every):
        th = v["theta"][t]
        if not np.isfinite(th).all():
            continue
        A = attention_scores(X[t], th)
        M = np.zeros_like(acc)
        np.add.at(M, (ids[:, None].repeat(len(ids), 1), ids[None, :].repeat(len(ids), 0)), A)
        counts = np.bincount(ids, minlength=len(names))
        acc += M / np.outer(counts, counts)
        n += 1
    return pd.DataFrame(acc / max(n, 1), index=names, columns=names)


def timing_comparison(models: dict, timing: pd.Series) -> dict:
    df = pd.DataFrame({"att": models["Linear attention"]["ret"], "timing": timing}).dropna()
    reg = ols_newey_west(df["att"].to_numpy()[:, None], df["timing"].to_numpy(), lags=3)
    resid = df["att"] - (reg["coef"][0, 0] + reg["coef"][1, 0] * df["timing"])
    return {"corr": float(df.corr().iloc[0, 1]), "r2": float(1 - resid.var() / df["att"].var()),
            "alpha_t": float(reg["t"][0, 0]), "timing_sharpe": _sharpe(df["timing"])}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _sub(sim, text):
    return text + ("  ·  SIMULATED DATA: tests the code only" if sim else "")


PALETTE = [plots.INK_3, plots.VIOLET, plots.AQUA, plots.YELLOW, plots.BLUE, plots.ORANGE, plots.RED, plots.GREEN]


def fig_growth(sc, out, sim):
    fig, ax, plt = plots.new_figure(9.5, 5.0)
    df = sc["returns"]
    target = 0.10 / math.sqrt(12)
    for k, col in zip(df.columns, PALETTE):
        r = df[k] / df[k].std() * target
        ax.plot(df.index.to_timestamp(), (1 + r).cumprod(), color=col, linewidth=1.6 if "attention" in k.lower() else 1.1,
                label=k)
    ax.set_yscale("log")
    ax.set_ylabel("Growth of $1 at 10% volatility (log scale)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    plots.titles(fig, "Lab Figure 22.1  Out-of-sample SDF portfolios, scaled to the same volatility",
                 _sub(sim, "Each model re-estimated every month on the previous 60 months"))
    plots.finish(fig, out / "fig22_1_growth.png")
    plt.close(fig)


def fig_sharpe(sc, out, sim, split):
    fig, ax, plt = plots.new_figure(9.0, 4.6)
    t = sc["table"]
    y = np.arange(len(t))
    ax.barh(y + 0.2, t["sharpe"], 0.4, color=plots.INK_3, label="Full period")
    ax.barh(y - 0.2, t[f"sharpe_from_{split[:4]}"], 0.4, color=plots.BLUE, label=f"From {split[:4]}")
    ax.set_yticks(y, t["model"])
    ax.invert_yaxis()
    ax.axvline(0, color=plots.INK, linewidth=0.8)
    ax.set_xlabel("Out-of-sample Sharpe ratio (annualized)")
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    plots.titles(fig, "Lab Figure 22.2  Sharpe ratio by model",
                 _sub(sim, "Transformer rows appear when PyTorch is installed (Part D), one row per number of blocks"))
    plots.finish(fig, out / "fig22_2_sharpe.png")
    plt.close(fig)


def fig_attention(att, out, sim):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    fig.patch.set_facecolor(plots.SURFACE)
    v = att.to_numpy()
    m = np.nanmax(np.abs(v)) or 1
    cmap = LinearSegmentedColormap.from_list("div", [plots.BLUE, "#f0efec", plots.RED])
    ax.imshow(v, cmap=cmap, norm=TwoSlopeNorm(vmin=-m, vcenter=0, vmax=m))
    ax.set_xticks(range(len(att)), att.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(att)), att.index, fontsize=8)
    ax.set_xlabel("Cluster whose characteristics are used", color=plots.INK_2)
    ax.set_ylabel("Cluster whose weight is set", color=plots.INK_2)
    for s in ax.spines.values():
        s.set_visible(False)
    plots.titles(fig, "Lab Figure 22.3  Average attention between factor clusters",
                 _sub(sim, "Linear-attention model · red: a cluster's characteristics raise the other's weight; blue: "
                           "lower it"))
    plots.finish(fig, out / "fig22_3_attention.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _md(df, fmt):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt[c](r[c]) if c in fmt else str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def _num(spec):
    return lambda v: "" if (v is None or (isinstance(v, float) and math.isnan(v))) else format(v, spec)


def write_report(out, p, sc, cst, rob, att, tim, torch_ok, sim, split, args):
    md = ["# Lab 22 results: Does Attention Across Assets Price Returns Better?", ""]
    if sim:
        md += ["> **Simulated data.** These results test that the code runs; they say nothing about real markets.", ""]
    md += ["**Claim on trial:** attention across assets improves an SDF's out-of-sample pricing and Sharpe ratio.", "",
           f"Assets: {p['R'].shape[1]} JKP factors, {p['dates'][0]} to {p['dates'][-1]}; window {args.window} months.",
           "", "## Parts A to C and E: models and scores", "",
           _md(sc["table"], {c: _num(".2f") for c in sc["table"].columns if "sharpe" in c} |
               {c: _num(".2%") for c in sc["table"].columns if "pricing" in c}), "",
           "Pricing error: root-mean-square alpha of the factors' next-month returns on the SDF portfolio, per year.", "",
           "**Alpha t-statistics of each row's model against each column's model**", "",
           _md(sc["alpha_t"].reset_index().rename(columns={"index": "model"}),
               {c: _num("+.2f") for c in sc["alpha_t"].columns}), "",
           "![Lab Figure 22.1](fig22_1_growth.png)", "", "![Lab Figure 22.2](fig22_2_sharpe.png)", ""]
    md += ["## Part D: a small transformer", "",
           "Included in the tables above." if torch_ok else
           "Skipped: PyTorch is not installed (`pip install torch`). Parts A–C and E do not need it.", ""]
    md += ["## Robustness", "",
           f"**Costs** ({args.cost * 1e4:.0f} basis points per unit of change in the weights on the factors; this "
           "understates the true cost, because each factor is itself a long–short portfolio):", "",
           _md(cst, {"avg_weight_change": _num(".2f"), "sharpe_gross": _num(".2f"), "sharpe_net": _num(".2f")}), "",
           "**Placebo and window length** (linear attention):", "",
           _md(rob, {"sharpe": _num(".2f")}), "",
           f"**A simpler explanation:** correlation of linear attention with a factor-timing rule (high past return, "
           f"low volatility) {tim['corr']:.2f}; R² {tim['r2']:.2f}; alpha t after controlling for it "
           f"{tim['alpha_t']:.2f}; the rule's own Sharpe ratio {tim['timing_sharpe']:.2f}.", "",
           "![Lab Figure 22.3](fig22_3_attention.png)", ""]
    md += ["## Your verdict", "",
           "1. Does attention add a significant alpha over the best model without attention, out of sample and after "
           f"{split[:4]}?",
           "2. Does the advantage survive the placebo, costs and a 120-month window?",
           "3. How much of it does a simple factor-timing rule explain?", ""]
    (out / "lab22_report.md").write_text("\n".join(md))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["files", "sim"], default="files")
    ap.add_argument("--jkp-file", default=None, help="JKP U.S. monthly factor returns (CSV)")
    ap.add_argument("--clusters-file", default=None, help="JKP cluster labels (CSV)")
    ap.add_argument("--weighting", default=None, help="JKP weighting scheme to keep, e.g. vw_cap")
    ap.add_argument("--ff5-file", default=None, help="Fama/French 5 Factors (2x3) zip or CSV")
    ap.add_argument("--mom-file", default=None, help="Momentum Factor (Mom) zip or CSV")
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--cost", type=float, default=0.001, help="cost per unit of weight change (decimal)")
    ap.add_argument("--blocks", type=int, nargs="*", default=[1, 2, 3, 4], help="Part D: transformer depths")
    ap.add_argument("--seeds", type=int, default=10, help="Part D: random starts per fit")
    ap.add_argument("--refit", type=int, default=12, help="Part D: months between refits")
    ap.add_argument("--no-transformer", action="store_true", help="skip Part D even if PyTorch is installed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sim-seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "report"))
    return ap.parse_args(argv)


def run(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = {"jkp": args.jkp_file, "clusters": args.clusters_file, "weighting": args.weighting, "ff5": args.ff5_file,
             "mom": args.mom_file}
    try:
        d = load(args.source, files, args.sim_seed)
    except DataUnavailable as e:
        raise SystemExit(str(e)) from None
    sim = d["simulated"]
    p = prepare(d)
    models = run_models(p, d["ff6"], args.window, seed=args.seed)
    extra, torch_ok = {}, False
    if not args.no_transformer:
        for L in args.blocks:
            r = transformer_sdf(p, blocks=L, window=args.window, refit=args.refit, seeds=args.seeds)
            if r is None:
                break
            torch_ok = True
            extra[f"Transformer, {L} block{'s' if L > 1 else ''}"] = r
    sc = score(models, p["R"], SPLIT, extra)
    cst = costs(models, p, args.cost)
    plac = run_models(p, d["ff6"], args.window, seed=args.seed, placebo=True)
    long_w = walk_forward_sdf(models["Linear attention"]["F"], 120)
    rob = pd.DataFrame([
        {"variant": "Linear attention, 60-month window", "sharpe": _sharpe(models["Linear attention"]["ret"])},
        {"variant": "Placebo: characteristics shuffled across factors", "sharpe": _sharpe(plac["Linear attention"]["ret"])},
        {"variant": "Linear attention, 120-month window", "sharpe": _sharpe(pd.Series(long_w["ret"], index=p["dates"]))},
    ])
    tim = timing_comparison(models, timing_rule(p))
    att = attention_by_cluster(models, p)
    fig_growth(sc, out, sim)
    fig_sharpe(sc, out, sim, SPLIT)
    fig_attention(att, out, sim)
    sc["table"].to_csv(out / "table_scores.csv", index=False)
    sc["alpha_t"].to_csv(out / "table_alpha_t.csv")
    sc["alpha"].to_csv(out / "table_alpha_annual.csv")
    cst.to_csv(out / "table_costs.csv", index=False)
    rob.to_csv(out / "table_robustness.csv", index=False)
    att.to_csv(out / "table_attention_clusters.csv")
    write_report(out, p, sc, cst, rob, att, tim, torch_ok, sim, SPLIT, args)
    summary = {"simulated": sim, "scores": sc["table"].to_dict(orient="records"), "costs": cst.to_dict(orient="records"),
               "robustness": rob.to_dict(orient="records"), "timing": tim, "transformer": torch_ok}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    for _, r in sc["table"].iterrows():
        print(f"{r.model:32s} Sharpe {r.sharpe:5.2f}  pricing error {r.pricing_error:.2%}")
    if not torch_ok and not args.no_transformer:
        print("Part D skipped: PyTorch is not installed.")
    print(f"Placebo Sharpe {rob.sharpe.iloc[1]:.2f}; correlation with the timing rule {tim['corr']:.2f}")
    print(f"Report: {out / 'lab22_report.md'}")
    if sim:
        print("NOTE: simulated data. These results test the code only.")
    return summary


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
