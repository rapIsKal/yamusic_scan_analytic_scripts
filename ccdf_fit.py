"""
CCDF Fit: y = C * x^a * exp(-(x/b)^lambda)
--------------------------------------------
Fits a generalized gamma survival function to the empirical CCDF
of monthly listeners. Fitting and R² are computed on log(y) so
every decade of the tail is weighted equally.

Usage:
    python ccdf_fit.py --db your.db [--out ccdf.png]
                       [--min-listeners 2] [--max-listeners 1000000]
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def r2_log(y_actual, y_predicted):
    """R² on log(y) — equal weight across all decades."""
    la = np.log(np.clip(y_actual,    1e-12, None))
    lp = np.log(np.clip(y_predicted, 1e-12, None))
    ss_res = np.sum((la - lp) ** 2)
    ss_tot = np.sum((la - la.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot else 0.0


def gen_gamma(x, C, a, b, lam):
    return C * np.power(x, a) * np.exp(-np.power(x / b, lam))


def gen_gamma_log(x, logC, a, b, lam):
    """log(y) form for fitting — avoids overflow, numerically stable."""
    return logC + a * np.log(x) - np.power(x / b, lam)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db",  required=True)
    parser.add_argument("--out", default="ccdf.png")
    parser.add_argument("--min-listeners", type=int, default=2,
                        help="Exclude artists with <= this many listeners (default 2)")
    parser.add_argument("--max-listeners", type=int, default=None,
                        help="Exclude artists with >= this many listeners")
    args = parser.parse_args()

    if not Path(args.db).exists():
        sys.exit(f"[ERROR] DB not found: {args.db}")

    # ── Load ─────────────────────────────────────────────────────────────────
    con = sqlite3.connect(args.db)
    query  = "SELECT monthly_listeners FROM artists WHERE monthly_listeners > ?"
    params = [args.min_listeners]
    if args.max_listeners is not None:
        query += " AND monthly_listeners < ?"
        params.append(args.max_listeners)
    listeners = np.array([v for (v,) in con.execute(query, params)], dtype=float)
    con.close()

    if len(listeners) == 0:
        sys.exit("[ERROR] No data after filtering.")
    print(f"[INFO] {len(listeners):,} artists  |  "
          f"range: {listeners.min():.0f} – {listeners.max():.0f}")

    # ── Empirical CCDF ────────────────────────────────────────────────────────
    sv      = np.sort(listeners)
    n       = len(sv)
    ccdf    = (n - np.arange(1, n + 1)).astype(float)

    # Drop trailing zeros (log undefined) and x=0
    mask = (ccdf > 0) & (sv > 0)
    sv_nz, ccdf_nz = sv[mask], ccdf[mask]

    # ── Log-spaced subsample so every decade gets equal fit points ────────────
    MAX_FIT = 2000
    if len(sv_nz) > MAX_FIT:
        grid = np.linspace(np.log(sv_nz[0]), np.log(sv_nz[-1]), MAX_FIT)
        idx  = np.unique(np.searchsorted(sv_nz, np.exp(grid)).clip(0, len(sv_nz)-1))
        x_fit, y_fit = sv_nz[idx], ccdf_nz[idx]
    else:
        x_fit, y_fit = sv_nz, ccdf_nz

    log_y_fit = np.log(y_fit)

    # Debug: points per decade
    print(f"[INFO] {len(x_fit)} fit points across decades:")
    d = x_fit.min()
    while d < x_fit.max():
        c = int(np.sum((x_fit >= d) & (x_fit < d * 10)))
        print(f"       {d:>10,.0f} – {d*10:>10,.0f}: {c} points")
        d *= 10

    # ── Fit: log(y) = log(C) + a*log(x) - (x/b)^lam ─────────────────────────
    try:
        p, _ = curve_fit(
            gen_gamma_log, x_fit, log_y_fit,
            p0=[np.log(y_fit.max()), -0.3, x_fit.mean(), 0.5],
            bounds=([-np.inf, -np.inf, 1e-6, 1e-6],
                    [ np.inf,  np.inf, np.inf, np.inf]),
            maxfev=200000,
        )
        logC, a, b, lam = p
        C     = np.exp(logC)
        r2_gg = r2_log(y_fit, gen_gamma(x_fit, C, a, b, lam))
        fit_ok = True
        print(f"\n── Fit: y = C·x^a·e^(-(x/b)^λ) ────────────────────")
        print(f"   C   = {C:.4g}")
        print(f"   a   = {a:.4f}  (shape of initial rise/drop)")
        print(f"   b   = {b:.4g}  (scale)")
        print(f"   λ   = {lam:.4f}  (λ<1 → heavy tail, λ=1 → exponential)")
        print(f"   R²  = {r2_gg:.4f}  (on log y)")
    except RuntimeError as e:
        fit_ok = False
        print(f"[WARN] Fit did not converge: {e}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_yscale("log")
    ax.plot(sv_nz, ccdf_nz, color="#4C72B0", linewidth=1.5, label="Empirical CCDF")

    if fit_ok:
        x_line = np.linspace(sv_nz[0], sv_nz[-1], 2000)
        y_line = gen_gamma(x_line, C, a, b, lam)
        ax.plot(x_line[y_line > 0], y_line[y_line > 0],
                color="crimson", linewidth=2.5, linestyle="--",
                label=rf"$C \cdot x^a \cdot e^{{-(x/b)^\lambda}}$   R²={r2_gg:.4f}")

        annotation = (
            f"$y = C \\cdot x^a \\cdot e^{{-(x/b)^\\lambda}}$\n"
            f"$C={C:.3g},\\ a={a:.3f}$\n"
            f"$b={b:.3g},\\ \\lambda={lam:.3f}$\n"
            f"$R^2={r2_gg:.4f}$ (on $\\log y$)"
        )
        ax.annotate(annotation,
                    xy=(0.97, 0.95), xycoords="axes fraction",
                    ha="right", va="top", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.9))

    ax.set_xlabel("N  (monthly listeners)", fontsize=11)
    ax.set_ylabel("Artists with > N listeners  (log scale)", fontsize=11)
    ax.set_title(r"CCDF fit:  $y = C \cdot x^a \cdot e^{-(x/b)^\lambda}$",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\n[OK] Chart saved → {args.out}")
    plt.show()


if __name__ == "__main__":
    main()