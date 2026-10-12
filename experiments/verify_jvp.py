from dataclasses import replace
from pathlib import Path
import argparse, sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from huggett_nesvm_core import Params
from huggett_exponential_collocation import residual_stage_shared, residual_stage_jvp_shared
import huggett_nesvm_core as core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--I", type=int, default=60)
    ap.add_argument("--N", type=int, default=12)
    ap.add_argument("--T", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    p0 = Params(I=args.I, lambda2=0.5)
    pT = replace(p0, lambda2=0.8)
    e0 = core.stationary_equilibrium(p0)
    eT = core.stationary_equilibrium(pT)
    r = np.full(args.N, eT["r"])
    f0, base = residual_stage_shared(r, pT, e0, eT, args.T)

    rng = np.random.default_rng(args.seed)
    d = rng.standard_normal(args.N)
    d /= np.linalg.norm(d)
    jv = residual_stage_jvp_shared(r, d, pT, e0, eT, args.T, base=base)

    rows = []
    for eps in (1e-4, 3e-5, 1e-5, 3e-6, 1e-6):
        fp, _ = residual_stage_shared(r + eps * d, pT, e0, eT, args.T)
        fd = (fp - f0) / eps
        rows.append({
            "eps": eps,
            "rel_error": np.linalg.norm(fd - jv) / max(np.linalg.norm(fd), 1e-14),
            "max_abs_error": np.max(np.abs(fd - jv)),
        })

    out = pd.DataFrame(rows)
    outdir = ROOT / "results" / "generated"
    outdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(outdir / "shared_stage_jvp_verification.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
