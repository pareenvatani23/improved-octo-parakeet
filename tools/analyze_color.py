#!/usr/bin/env python3
"""Does app ICON COLOR relate to installs — and does it survive controlling for
category and age? Joins data/icon_colors.csv to data/googleplay_cohort.csv.

Honest structure:
  1) raw: install hit-rate by color (confounded).
  2) confound check: color composition across top categories.
  3) the real test: out-of-time AUC for [genre+year] vs [genre+year+color] vs
     [color only]. If color adds ~0 over genre+year, there is no color signal.

Caveat (see PILOT_RESULTS.md): the cohort is survivor/age-biased, so this is
descriptive/exploratory, not a validated predictor.
"""
import sys, re
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

COH = Path("data/googleplay_cohort.csv")
COL = Path("data/icon_colors.csv")


def read_csv_skip(path):
    lines = [l for l in open(path, encoding="utf-8") if not l.startswith("#")]
    from io import StringIO
    return pd.read_csv(StringIO("".join(lines)))


def parse_date(s):
    s = str(s or "")
    for fmt in ("%b %d, %Y", "%d %b %Y", "%B %d, %Y"):
        try: return datetime.strptime(s, fmt)
        except ValueError: pass
    m = re.search(r"(\d{4})", s)
    return datetime(int(m.group(1)), 1, 1) if m else None


def top_lift(y, s, base, frac=0.10):
    n = max(1, int(len(s) * frac)); idx = np.argsort(s)[::-1][:n]
    return (float(np.mean(y[idx])) / base) if base > 0 else float("nan")


def main():
    if not COH.exists() or not COL.exists():
        print("need both data/googleplay_cohort.csv and data/icon_colors.csv", file=sys.stderr); sys.exit(1)
    coh = read_csv_skip(COH); col = read_csv_skip(COL)
    df = coh.merge(col[["appId", "color", "sat", "val", "hue"]], on="appId", how="inner")
    df["minInstalls"] = pd.to_numeric(df["minInstalls"], errors="coerce").fillna(0)
    df["date"] = df["released"].map(parse_date)
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    cutoff = df["minInstalls"].quantile(0.90)
    df["hit"] = (df["minInstalls"] >= cutoff).astype(int)
    print(f"joined apps: {len(df)} | install-hit cutoff (top decile): {int(cutoff):,} | base rate {df.hit.mean()*100:.1f}%\n")

    # 1) raw hit-rate by color
    print("=== RAW install-hit rate by icon color (CONFOUNDED) ===")
    g = df.groupby("color").agg(n=("hit", "size"), hit_rate=("hit", "mean"),
                                median_installs=("minInstalls", "median")).sort_values("hit_rate", ascending=False)
    for c, row in g.iterrows():
        print(f"  {c:<8} n={int(row.n):<4} hit-rate {row.hit_rate*100:4.1f}%  median installs {int(row.median_installs):,}")

    # 2) confound: color mix within top categories
    print("\n=== color composition within top categories (why raw is misleading) ===")
    for genre in df["genre"].value_counts().head(4).index:
        sub = df[df.genre == genre]
        top = sub["color"].value_counts(normalize=True).head(3)
        print(f"  {genre:<18} " + ", ".join(f"{c} {p*100:.0f}%" for c, p in top.items()))

    # 3) the real test: out-of-time AUC, color beyond genre+year
    df["year"] = df["date"].dt.year.astype(str)
    cut = int(len(df) * 0.70); tr, te = df.iloc[:cut], df.iloc[cut:]
    base = te.hit.mean(); yte = te.hit.values.astype(int)
    if base in (0, 1) or len(np.unique(yte)) < 2:
        print("\n(test fold has degenerate labels — cohort too age-skewed for a clean split)")
        return

    def auc(cols):
        cat = ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), cols)])
        m = Pipeline([("p", cat), ("l", LogisticRegression(max_iter=1000, class_weight="balanced"))])
        m.fit(tr[cols], tr.hit.values); p = m.predict_proba(te[cols])[:, 1]
        return roc_auc_score(yte, p), top_lift(yte, p, base)

    print(f"\n=== OUT-OF-TIME test (base rate {base*100:.1f}%) ===")
    for label, cols in [("genre + year (baseline)", ["genre", "year"]),
                        ("genre + year + COLOR", ["genre", "year", "color"]),
                        ("COLOR only", ["color"])]:
        a, l = auc(cols)
        print(f"  {label:<26} ROC {a:.3f}  top-decile lift {l:.2f}x")

    print("\nVERDICT: if 'genre+year+color' ROC ≈ 'genre+year', color adds nothing beyond "
          "category/age. Reminder: survivor/age-biased cohort → descriptive only.")


if __name__ == "__main__":
    main()
