#!/usr/bin/env python3
"""Train + HONESTLY evaluate an outcome model on data/features.csv.

The whole point is the honesty gate:
  - TIME-SPLIT: train on earlier launches, test on strictly later ones
    (in-sample fit is meaningless; out-of-time lift is the only truth).
  - Metrics: PR-AUC (classes imbalanced), ROC-AUC, and top-decile precision &
    LIFT vs the base rate ("of the ideas it ranked top-10%, how many were hits?").
  - Baselines to beat: random (= base rate) and word_count-only.
  - Kill criterion printed: if out-of-time lift ~1x, we say so.

Usage: python tools/train_eval.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

FEAT = Path("data/features.csv")
TAXO_COLS = ["trend", "audience", "mechanic", "loop", "money", "format"]
CANDIDATE_NUM = ["has_ai", "word_count", "free", "offersIAP"]


def topk_precision_lift(y_true, scores, base, frac=0.10):
    n = max(1, int(len(scores) * frac))
    idx = np.argsort(scores)[::-1][:n]
    prec = float(np.mean(y_true[idx])) if n else 0.0
    lift = prec / base if base > 0 else float("nan")
    return prec, lift, n


def main():
    if not FEAT.exists():
        print(f"missing {FEAT} — run extract_features.py first", file=sys.stderr); sys.exit(1)
    df = pd.read_csv(FEAT)
    df = df.sort_values("created").reset_index(drop=True)
    NUM_COLS = [c for c in CANDIDATE_NUM if c in df.columns]
    n = len(df)
    print(f"cohort: {n} apps | hits: {int(df.hit.sum())} ({df.hit.mean()*100:.1f}%)")
    if df.hit.sum() < 8 or n < 60:
        print("WARNING: too few apps/hits for a trustworthy split; results are indicative only.")

    # time split: train on earliest 70%, test on latest 30%
    cut = int(n * 0.70)
    train, test = df.iloc[:cut], df.iloc[cut:]
    print(f"time-split: train {len(train)} (< {train.created.iloc[-1]})  |  test {len(test)} (>= {test.created.iloc[0]})")
    base = test.hit.mean()
    print(f"test base rate (random baseline) = {base*100:.1f}%")
    if base == 0 or base == 1:
        print("degenerate test labels; aborting eval."); sys.exit(0)

    Xtr, ytr = train[TAXO_COLS + NUM_COLS], train.hit.values
    Xte, yte = test[TAXO_COLS + NUM_COLS], test.hit.values.astype(int)

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), TAXO_COLS),
        ("num", "passthrough", NUM_COLS),
    ])

    models = {
        "logistic": Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
        "grad_boost": Pipeline([("pre", pre), ("clf", GradientBoostingClassifier(random_state=0))]),
    }

    print("\n=== OUT-OF-TIME RESULTS (the honest numbers) ===")
    print(f"{'model':<12} {'PR-AUC':>7} {'ROC-AUC':>8} {'top10%prec':>11} {'lift':>6}")
    results = {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        prauc = average_precision_score(yte, p)
        rocauc = roc_auc_score(yte, p) if len(np.unique(yte)) > 1 else float("nan")
        prec, lift, k = topk_precision_lift(yte, p, base)
        results[name] = (prauc, rocauc, prec, lift)
        print(f"{name:<12} {prauc:>7.3f} {rocauc:>8.3f} {prec*100:>10.1f}% {lift:>5.2f}x")

    # baseline: word_count only
    wc = test.word_count.values.astype(float)
    _, lift_wc, _ = topk_precision_lift(yte, wc, base)
    print(f"{'wordcount':<12} {'-':>7} {'-':>8} {'-':>11} {lift_wc:>5.2f}x   (dumb baseline)")

    # ablation: does the IDEA TAXONOMY carry signal WITHOUT the length/other confounders?
    taxo_pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), TAXO_COLS)])
    taxo_m = Pipeline([("pre", taxo_pre), ("clf", GradientBoostingClassifier(random_state=0))])
    taxo_m.fit(train[TAXO_COLS], ytr)
    tp = taxo_m.predict_proba(test[TAXO_COLS])[:, 1]
    taxo_roc = roc_auc_score(yte, tp) if len(np.unique(yte)) > 1 else float("nan")
    _, taxo_lift, _ = topk_precision_lift(yte, tp, base)
    print(f"{'TAXONOMY-only':<12} {'-':>7} {taxo_roc:>8.3f} {'-':>11} {taxo_lift:>5.2f}x   (the real question)")

    # interpretability: gradient-boost feature importances
    gb = models["grad_boost"]
    names = gb.named_steps["pre"].get_feature_names_out()
    imp = gb.named_steps["clf"].feature_importances_
    order = np.argsort(imp)[::-1][:10]
    print("\ntop learned signals (gradient boosting importances):")
    for i in order:
        if imp[i] > 0:
            print(f"  {names[i]:<28} {imp[i]:.3f}")

    # verdict
    best_lift = max(results[m][3] for m in results)
    best_prauc = max(results[m][0] for m in results)
    print("\n=== VERDICT ===")
    print(f"best out-of-time top-decile lift = {best_lift:.2f}x  (PR-AUC {best_prauc:.3f} vs base {base:.3f})")
    if best_lift >= 1.5 and best_prauc > base * 1.15:
        print("SIGNAL: the model ranks meaningfully better than chance out-of-time.")
    elif best_lift >= 1.2:
        print("WEAK SIGNAL: modestly better than chance; treat as directional only.")
    else:
        print("NO RELIABLE SIGNAL: not better than chance out-of-time. Honest result — "
              "buzz on this data isn't predictable from these features (expected for a weak label).")
    print("\nNote: label = Product Hunt votes (a BUZZ proxy, not revenue). This tests the "
          "pipeline and whether ANY signal exists — not commercial success.")


if __name__ == "__main__":
    main()
