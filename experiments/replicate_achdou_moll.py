from dataclasses import replace
from pathlib import Path
from time import perf_counter
import argparse
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import huggett_nesvm_core as core


def achdou_transition(I=1000, T=20.0, N=100, tol=1e-5, maxit=1000, verbose=True):
    p0 = core.Params(I=I, lambda2=0.5, stationary_tol=1e-9)
    pT = replace(p0, lambda2=0.8)
    e0 = core.stationary_equilibrium(p0)
    eT = core.stationary_equilibrium(pT)
    a = eT["a"]
    ns = 2 * I
    dt = T / N
    eye = sp.eye(ns, format="csc")
    r = np.full(N, eT["r"])
    xi = 20 * (np.exp(-0.05 * np.arange(1, N + 1)) - np.exp(-0.05 * N))
    V = np.empty((ns, N + 1))
    G = np.empty((ns, N + 1))
    ops = [None] * N
    history = []
    t0 = perf_counter()

    for it in range(maxit):
        V[:, -1] = eT["v"]
        v = eT["v"].copy()
        for n in range(N - 1, -1, -1):
            A, u, _ = core.build_generator(v, a, float(r[n]), pT)
            v = spla.spsolve((pT.rho + 1 / dt) * eye - A, u + v / dt)
            V[:, n] = v
            ops[n] = A

        G[:, 0] = e0["g"]
        assets = np.empty(N + 1)
        assets[0] = np.r_[a, a] @ G[:, 0]
        for n in range(N):
            G[:, n + 1] = spla.spsolve(eye - ops[n].T * dt, G[:, n])
            assets[n + 1] = np.r_[a, a] @ G[:, n + 1]

        dS = np.diff(assets)
        err = float(np.max(np.abs(dS)))
        history.append(err)
        if verbose and (it < 5 or (it + 1) % 10 == 0 or err < tol):
            print(f"iter {it+1}: err={err:.3e}, r=[{r.min():.6f},{r.max():.6f}]")
        if err < tol:
            break
        r = r - xi * dS

    return {
        "time": np.arange(1, N + 1) * dt,
        "r": r,
        "history": np.asarray(history),
        "e0": e0,
        "eT": eT,
        "assets": assets,
        "G": G,
        "V": V,
        "seconds": perf_counter() - t0,
        "converged": history[-1] < tol,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--I", type=int, default=1000)
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--T", type=float, default=20.0)
    ap.add_argument("--tol", type=float, default=1e-5)
    args = ap.parse_args()
    out = achdou_transition(I=args.I, N=args.N, T=args.T, tol=args.tol)
    outdir = ROOT / "results" / "generated"
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"time_year": out["time"], "r_achdou": out["r"]}).to_csv(
        outdir / f"achdou_reference_I{args.I}_N{args.N}.csv", index=False
    )
    pd.DataFrame({"iteration": np.arange(1, len(out["history"]) + 1), "max_dS": out["history"]}).to_csv(
        outdir / f"achdou_reference_I{args.I}_N{args.N}_history.csv", index=False
    )
    print("stationary", out["e0"]["r"], out["eT"]["r"], "iters", len(out["history"]),
          "sec", out["seconds"], "conv", out["converged"])


if __name__ == "__main__":
    main()
