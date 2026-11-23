#!/usr/bin/env python3
"""
QDA aircraft-type classifier for the KCOS departure data.

    pip install pandas numpy scikit-learn matplotlib
    python qda_aircraft.py                       # uses train.csv / test.csv in this folder
    python qda_aircraft.py --plot-x gs_kt --plot-y alt_ft --min-points 300

Produces one window / qda_results.png with two panels:
  1. The model: a 2-feature QDA's decision regions (background colours) over the
     training points, so you can see how QDA carves up feature space.
  2. Accuracy: confusion matrix of the full-feature QDA on the TEST set
     (rows = true type, columns = predicted type, each row normalised).

Also prints point-level accuracy and a per-flight (majority-vote) accuracy.

Notes
-----
* Train and test were split by aircraft, so no aircraft appears in both.
* Features per point: altitude, ground speed, vertical rate (filled in from
  altitude changes where missing), distance from KCOS and turn rate.
* QDA fits one covariance matrix per class, so a class needs well more points
  than features. Types with fewer than --min-points training points are
  dropped (and test points of those types are dropped too, and reported).
* reg_param shrinks each covariance toward identity; it keeps QDA stable when
  features are correlated (altitude and distance from the airport are).
"""
import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

KCOS = (38.80581, -104.70076)
FEATURES = ["alt_ft", "gs_kt", "vrate_fpm", "dist_km", "turn_rate"]


def haversine_km(lat, lon, lat0, lon0):
    p1, p2 = np.radians(lat), math.radians(lat0)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * math.cos(p2) * np.sin(np.radians(lon0 - lon) / 2) ** 2)
    return 12742 * np.arcsin(np.sqrt(a))


def add_features(df):
    df = df.sort_values(["icao", "leg", "t_unix"]).copy()
    g = df.groupby(["icao", "leg"], sort=False)
    dt = g["t_unix"].diff()

    # vertical rate: use reported value, else estimate from altitude change
    est_vr = g["alt_ft"].diff() / dt * 60
    df["vrate_fpm"] = df["vrate_fpm"].fillna(est_vr)

    # turn rate (deg/s), track difference wrapped to [-180, 180]
    dtrack = (g["track"].diff() + 180) % 360 - 180
    df["turn_rate"] = (dtrack / dt).clip(-20, 20)

    df["dist_km"] = haversine_km(df["lat"].values, df["lon"].values, *KCOS)

    # leg-first points have no diff; treat as level / not turning
    df["vrate_fpm"] = df["vrate_fpm"].fillna(0)
    df["turn_rate"] = df["turn_rate"].fillna(0)
    return df.dropna(subset=["type", "alt_ft", "gs_kt"])


def load(path):
    return add_features(pd.read_csv(path))


def plot_regions(ax, train, feat_x, feat_y, classes, cmap, reg):
    """Fit a 2-feature QDA and draw its decision regions + training points."""
    sc = StandardScaler().fit(train[[feat_x, feat_y]])
    X = sc.transform(train[[feat_x, feat_y]])
    y = np.searchsorted(classes, train["type"])
    qda2 = QuadraticDiscriminantAnalysis(reg_param=reg).fit(X, y)

    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 400),
        np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 400))
    zz = qda2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=np.arange(-0.5, len(classes) + 0.5), cmap=cmap, alpha=0.35)

    # draw a thinned sample of training points so the plot stays readable
    idx = np.random.RandomState(0).choice(len(X), size=min(len(X), 6000), replace=False)
    ax.scatter(X[idx, 0], X[idx, 1], c=y[idx], cmap=cmap, s=6, edgecolors="none",
               vmin=-0.5, vmax=len(classes) - 0.5)
    # axis tick labels back in original units
    ax.set_xlabel(f"{feat_x} (standardised)")
    ax.set_ylabel(f"{feat_y} (standardised)")
    acc2 = qda2.score(X, y)
    ax.set_title(f"QDA decision regions on {feat_x} vs {feat_y}\n"
                 f"(2-feature model, train accuracy {acc2:.0%})")
    handles = [plt.Line2D([], [], marker="s", ls="", color=cmap(i), label=c)
               for i, c in enumerate(classes)]
    ax.legend(handles=handles, title="type", fontsize=7, ncol=2, loc="best")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="train.csv")
    ap.add_argument("--test", default="test.csv")
    ap.add_argument("--features", nargs="+", default=FEATURES, choices=FEATURES)
    ap.add_argument("--plot-x", default="gs_kt", choices=FEATURES)
    ap.add_argument("--plot-y", default="alt_ft", choices=FEATURES)
    ap.add_argument("--min-points", type=int, default=300,
                    help="drop aircraft types with fewer training points than this")
    ap.add_argument("--min-flights", type=int, default=3,
                    help="drop aircraft types with fewer training flights than this "
                         "(a type seen in one flight can't be learned in a way that generalises)")
    ap.add_argument("--reg", type=float, default=0.1, help="QDA reg_param (0-1)")
    ap.add_argument("--out", default="qda_results.png")
    ap.add_argument("--no-show", action="store_true", help="save the PNG but don't open a window")
    args = ap.parse_args()

    train, test = load(args.train), load(args.test)

    # keep only types with enough training points; test must use the same set
    counts = train["type"].value_counts()
    flights_per_type = train.drop_duplicates(["icao", "leg"])["type"].value_counts()
    ok = counts[(counts >= args.min_points)
                & (flights_per_type.reindex(counts.index).fillna(0) >= args.min_flights)]
    keep = sorted(ok.index)
    dropped = sorted(set(train["type"]) - set(keep))
    train = train[train["type"].isin(keep)]
    unseen = test[~test["type"].isin(keep)]
    test = test[test["type"].isin(keep)]
    if len(keep) < 2 or test.empty:
        raise SystemExit(f"Only {len(keep)} usable types / {len(test)} test points; "
                         "lower --min-points / --min-flights")
    print(f"Classes kept ({len(keep)}): {', '.join(keep)}")
    print(f"Dropped from training (< {args.min_points} points or < {args.min_flights} flights): "
          f"{', '.join(dropped) or 'none'}")
    print(f"Test points dropped (type not kept): {len(unseen)} "
          f"({', '.join(sorted(unseen['type'].unique())) or 'none'})")

    # ---- full-feature model --------------------------------------------- #
    scaler = StandardScaler().fit(train[args.features])
    enc = LabelEncoder().fit(keep)
    qda = QuadraticDiscriminantAnalysis(reg_param=args.reg)
    qda.fit(scaler.transform(train[args.features]), enc.transform(train["type"]))

    y_true = enc.transform(test["type"])
    y_pred = qda.predict(scaler.transform(test[args.features]))
    acc = accuracy_score(y_true, y_pred)
    print(f"\nTest accuracy (per point): {acc:.1%}  ({len(test):,} points)")

    # per-flight accuracy: majority vote of the point predictions in each leg
    flights = test.assign(pred=y_pred, true=y_true).groupby(["icao", "leg"])
    voted = flights["pred"].agg(lambda s: np.bincount(s).argmax())
    truth = flights["true"].first()
    print(f"Test accuracy (per flight, majority vote): "
          f"{(voted == truth).mean():.1%}  ({len(truth)} flights)")

    # ---- figure --------------------------------------------------------- #
    base = plt.get_cmap("tab20")
    cmap = ListedColormap([base(i % 20) for i in range(len(keep))])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    plot_regions(ax1, train, args.plot_x, args.plot_y, np.array(keep), cmap, args.reg)

    # rows: only types that actually occur in the test set; columns: every class
    # the model can predict (so wrong guesses at unexpected types are visible)
    row_ids = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=range(len(keep)), normalize="true")[row_ids]
    counts_cm = confusion_matrix(y_true, y_pred, labels=range(len(keep)))[row_ids]
    ax2.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax2.set_xticks(range(len(keep)), keep, rotation=60)
    ax2.set_yticks(range(len(row_ids)), [f"{keep[i]} (n={counts_cm[r].sum():,})"
                                         for r, i in enumerate(row_ids)])
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            if counts_cm[r, c]:
                ax2.text(c, r, f"{cm[r, c]:.0%}", ha="center", va="center", fontsize=9,
                         color="white" if cm[r, c] > 0.5 else "black")
    ax2.set_xlabel("Predicted type")
    ax2.set_ylabel("True type (test points)")
    ax2.set_title(f"Confusion matrix on test set (each row sums to 100%)\n"
                  f"per-point accuracy {acc:.1%}, {len(test):,} points, "
                  f"{len(truth)} flights")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"Saved {args.out}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
