from dataclasses import replace
from pathlib import Path
import argparse, sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from huggett_nesvm_core import Params
from huggett_exponential_collocation import smoothstep_lambda2_mid, solve_stage_shared, interpolate_stage_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", nargs="+", type=int, default=[50, 100, 200, 400, 800, 1600])
    ap.add_argument("--N", type=int, default=80)
    ap.add_argument("--T", type=float, default=20.0)
    ap.add_argument("--tol", type=float, default=5e-9)
    args = ap.parse_args()

    outdir = ROOT / "results" / "generated"
    outdir.mkdir(parents=True, exist_ok=True)
    lam = smoothstep_lambda2_mid(args.N, args.T)
    x0 = None
    old_t = None
    old_r = None
    summary = []

    for I in args.grids:
        p0 = Params(I=I, lambda2=0.5)
        pT = replace(p0, lambda2=0.8)
        if old_r is not None:
            new_t = (np.arange(args.N) + 0.5) * args.T / args.N
            x0 = interpolate_stage_path(old_t, old_r, new_t)
        sol = solve_stage_shared(p0, pT, args.T, args.N, x0=x0, lambda2_mid=lam, tol=args.tol)
        pd.DataFrame({
            "time_mid_year": sol["tmid"],
            "r_stage": sol["r"],
            "residual": sol["residual"],
        }).to_csv(outdir / f"smooth_GE_I{I}_N{args.N}.csv", index=False)
        summary.append({
            "I": I,
            "r_initial_stationary": sol["eq0"]["r"],
            "r_terminal_stationary": sol["eqT"]["r"],
            "max_equilibrium_residual": float(np.max(np.abs(sol["residual"]))),
            "mass_error": sol["transition"]["mass_error"],
            "min_mass": sol["transition"]["min_mass"],
            "max_corrector_iterations": int(sol["transition"]["picard_iters"].max()),
            "seconds": sol["seconds"],
        })
        old_t, old_r = sol["tmid"], sol["r"]
        print(summary[-1])

    pd.DataFrame(summary).to_csv(outdir / "spatial_refinement_summary.csv", index=False)


if __name__ == "__main__":
    main()
