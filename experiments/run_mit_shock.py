from dataclasses import replace
from pathlib import Path
import argparse, sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from huggett_nesvm_core import Params
from huggett_exponential_collocation import solve_stage_shared


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--I", type=int, default=400)
    ap.add_argument("--N", type=int, default=80)
    ap.add_argument("--T", type=float, default=20.0)
    ap.add_argument("--tol", type=float, default=5e-9)
    args = ap.parse_args()

    p0 = Params(I=args.I, lambda2=0.5)
    pT = replace(p0, lambda2=0.8)
    lam = np.full(args.N, 0.8)
    sol = solve_stage_shared(p0, pT, args.T, args.N, lambda2_mid=lam, tol=args.tol)

    outdir = ROOT / "results" / "generated"
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "time_mid_year": sol["tmid"],
        "r_stage": sol["r"],
        "residual": sol["residual"],
    }).to_csv(outdir / f"MIT_shock_I{args.I}_N{args.N}.csv", index=False)

    print("converged", sol["converged"], "max residual", abs(sol["residual"]).max(),
          "mass error", sol["transition"]["mass_error"], "min mass", sol["transition"]["min_mass"])


if __name__ == "__main__":
    main()
