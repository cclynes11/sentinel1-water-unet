"""The numerics of the analysis stage: proxy placement, the weighted OOD score, held-out-event tests.

The method (supervisor meeting, 23 Oct 2026):
1. For each proxy, take its distribution over the model's training data.
2. Place every test tile in that distribution: its per-proxy score.
3. Regress per-tile model performance on the per-proxy scores. The coefficients are the
   importance weights; the OOD score of a tile is the weighted sum of its per-proxy scores.

Two per-proxy scores are computed, because they answer different questions:
* position  = percentile of the tile in the training distribution, rescaled to [-1, 1]
              (-1 = below every training tile, 0 = the training median, +1 = above every one).
              Signed: "more built-up is worse" is a statement about position.
* distance  = |position|: how far from the training median, whichever side. This is the
              out-of-distribution score proper: "unlike the training data is worse".
Percentiles are used rather than z-scores (sigma) because several proxies are zero-inflated or
long-tailed, where a z-score mostly reflects a tiny training standard deviation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ------------------------------------------------------------------------------------ placement
def percentile_of(ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Mid-rank empirical percentile (0-100) of each x within ref; NaN stays NaN."""
    r = np.sort(ref[np.isfinite(ref)])
    x = np.asarray(x, float)
    out = 100.0 * (np.searchsorted(r, x, side="left") + np.searchsorted(r, x, side="right")) / 2.0 / r.size
    out[~np.isfinite(x)] = np.nan
    return out


def place(tiles: pd.DataFrame, ref: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    """Add pct__<proxy>, z__<proxy>, pos__<proxy> and dist__<proxy> for every proxy.

    ``columns`` maps a proxy name to its column in both tables (e.g. slope_mean -> slope_mean__tile)."""
    out = tiles.copy()
    for proxy, col in columns.items():
        r = ref[col].to_numpy(float)
        r = r[np.isfinite(r)]
        x = out[col].to_numpy(float)
        pct = percentile_of(r, x)
        out[f"pct__{proxy}"] = pct
        out[f"z__{proxy}"] = (x - r.mean()) / r.std(ddof=1)
        out[f"pos__{proxy}"] = (pct - 50.0) / 50.0
        out[f"dist__{proxy}"] = np.abs(pct - 50.0) / 50.0
    return out


def prune_collinear(ref: pd.DataFrame, columns: dict[str, str], max_corr: float) -> tuple[list[str], dict]:
    """Keep proxies in the order given, dropping any whose |Spearman rho| with one already kept
    exceeds ``max_corr`` in the training distribution. Returns (kept, {dropped: (kept twin, rho)})."""
    corr = ref[list(columns.values())].corr(method="spearman")
    kept: list[str] = []
    dropped = {}
    for proxy, col in columns.items():
        twin = next(((k, corr.loc[col, columns[k]]) for k in kept if abs(corr.loc[col, columns[k]]) > max_corr), None)
        if twin:
            dropped[proxy] = (twin[0], float(twin[1]))
        else:
            kept.append(proxy)
    return kept, dropped


# ------------------------------------------------------------------------------------ correlation
def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10 or np.ptp(a[m]) == 0 or np.ptp(b[m]) == 0:
        return float("nan")
    return float(np.corrcoef(pd.Series(a[m]).rank(), pd.Series(b[m]).rank())[0, 1])


def partial_spearman(a, b, control) -> float:
    """Spearman correlation of a and b after removing, in rank space, what ``control`` explains."""
    a, b, c = (np.asarray(v, float) for v in (a, b, control))
    m = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
    if m.sum() < 10:
        return float("nan")
    ra, rb, rc = (pd.Series(v[m]).rank().to_numpy() for v in (a, b, c))

    def resid(y):
        X = np.column_stack([np.ones_like(rc), rc])
        return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]

    ea, eb = resid(ra), resid(rb)
    if ea.std() == 0 or eb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ea, eb)[0, 1])


# ------------------------------------------------------------------------------------ regression
class Linear:
    """Least squares on standardised features (optional ridge penalty ``alpha``).

    ``weights`` are standardised coefficients: the change in the target for one training
    standard deviation of the feature, so they are comparable between features."""

    def __init__(self, alpha: float = 0.0):
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> Linear:
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd == 0] = 1.0
        Z = (X - self.mu) / self.sd
        self.intercept = float(y.mean())
        A = Z.T @ Z + self.alpha * len(y) * np.eye(Z.shape[1])
        self.weights = np.linalg.lstsq(A, Z.T @ (y - self.intercept), rcond=None)[0]
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """The weighted sum of standardised features (the OOD score when X holds proxy scores)."""
        return ((X - self.mu) / self.sd) @ self.weights

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.intercept + self.score(X)


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    ss = float(((y - y.mean()) ** 2).sum())
    return float(1 - ((y - pred) ** 2).sum() / ss) if ss > 0 else float("nan")


def spatial_blocks(df: pd.DataFrame, size: int = 8) -> np.ndarray:
    """Block label per tile: tiles within the same size x size neighbourhood of a scene share one.

    Neighbouring tiles are not independent (same river, same city), so resampling single tiles
    would understate the uncertainty of the weights. Resampling blocks is the usual remedy."""
    return (df["event"].astype(str) + "_" + (df["tile_row"] // size).astype(str) + "_"
            + (df["tile_col"] // size).astype(str)).to_numpy()


def bootstrap_weights(X: np.ndarray, y: np.ndarray, blocks: np.ndarray, n: int, seed: int = 0,
                      alpha: float = 0.0) -> np.ndarray:
    """(n, features) weights from refits on block-resampled tiles."""
    rng = np.random.default_rng(seed)
    labels, inverse = np.unique(blocks, return_inverse=True)
    members = [np.where(inverse == i)[0] for i in range(len(labels))]
    out = np.empty((n, X.shape[1]))
    for b in range(n):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(labels), len(labels))])
        out[b] = Linear(alpha).fit(X[idx], y[idx]).weights
    return out


def held_out_event_test(df: pd.DataFrame, feature_sets: dict[str, list[str]], target: str,
                        group_col: str = "event_group") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit on all events but one, predict the one left out; repeat for every event.

    Returns (scores per held-out event and model, out-of-fold predictions per tile). The
    Spearman correlation between predicted and observed error on the unseen event is the
    headline number: a ranking is what a reviewer deciding where to look needs."""
    cols = sorted({c for fs in feature_sets.values() for c in fs})
    d = df.dropna(subset=[target, *cols]).reset_index(drop=True)
    preds = d[["tile_id", "event", group_col, target]].copy()
    rows = []
    groups = sorted(d[group_col].unique())
    for held in groups:
        test = (d[group_col] == held).to_numpy()
        for name, fs in feature_sets.items():
            if test.sum() < 10 or (~test).sum() < 10:
                continue
            model = Linear().fit(d.loc[~test, fs].to_numpy(float), d.loc[~test, target].to_numpy(float))
            p = model.predict(d.loc[test, fs].to_numpy(float))
            preds.loc[test, f"pred__{name}"] = p
            y = d.loc[test, target].to_numpy(float)
            rows.append({"held_out": held, "model": name, "n_tiles": int(test.sum()), "n_train_tiles": int((~test).sum()),
                         "spearman": spearman(p, y), "r2": r2(y, p)})
    return pd.DataFrame(rows), preds


# ------------------------------------------------------------------------------------ triage
def triage_curve(score: np.ndarray, error_px: np.ndarray, points: int = 101) -> tuple[np.ndarray, np.ndarray]:
    """Share of all error pixels found (y) when the top x share of tiles, ranked by ``score``, is reviewed."""
    m = np.isfinite(score) & np.isfinite(error_px)
    order = np.argsort(-score[m], kind="stable")
    cum = np.concatenate([[0.0], np.cumsum(error_px[m][order])])
    total = cum[-1] if cum[-1] > 0 else 1.0
    x = np.linspace(0, 1, points)
    return x, np.interp(x * m.sum(), np.arange(m.sum() + 1), cum) / total


def found_at(score: np.ndarray, error_px: np.ndarray, share: float = 0.2) -> float:
    x, y = triage_curve(score, error_px, 1001)
    return float(np.interp(share, x, y))
