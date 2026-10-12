# Structure-Preserving Exponential Collocation for Continuous-Time Heterogeneous-Agent Equilibria

Research code accompanying the manuscript by Braden Holst and Ralph E. White on a stage-centered exponential collocation method for continuous-time Huggett equilibrium transitions.

## What is in this repository

- `src/huggett_nesvm_core.py` — monotone/upwind Huggett discretization, stationary equilibrium routines, exponential affine steps, and Fréchet actions.
- `src/huggett_exponential_collocation.py` — final publication-candidate shared-stage HJB/Kolmogorov DAE solver. The same converged midpoint Markov generator is used in both backward HJB and forward Kolmogorov propagation.
- `experiments/replicate_achdou_moll.py` — independent Python reproduction of the canonical Achdou–Moll finite-difference Huggett MIT-shock transition benchmark.
- `experiments/run_smooth_ge.py` — smooth general-equilibrium transition using the final shared-stage solver.
- `experiments/run_mit_shock.py` — abrupt MIT-shock transition using the final shared-stage solver.
- `experiments/run_spatial_refinement.py` — spatial-refinement driver for the final shared-stage solver.
- `experiments/verify_jvp.py` — finite-difference verification of the Fréchet/Jacobian-vector product.
- `validation/` — validation helper scripts retained for paper reproducibility.
- `results/reference/` — canonical reference benchmark outputs used in the manuscript.
- `results/paper/` — selected manuscript-result tables and figures.
- `archive/development/` — earlier prototypes and intermediate research scripts retained for provenance. Some archived scripts contain historical absolute paths or superseded dependencies and are **not** the recommended entry points.
- `docs/` — supplementary manuscript material.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -e .
```

Alternatively:

```bash
pip install -r requirements.txt
```

## Reproduce the canonical Achdou–Moll benchmark

```bash
python experiments/replicate_achdou_moll.py --I 1000 --N 100 --T 20
```

This reproduces the public finite-difference transition algorithm for the benchmark parameterization used in the manuscript and writes the rate path and outer-iteration history to `results/generated/`.

## Run the final shared-stage solver

Smooth transition:

```bash
python experiments/run_smooth_ge.py --I 400 --N 80 --T 20
```

Abrupt MIT shock:

```bash
python experiments/run_mit_shock.py --I 400 --N 80 --T 20
```

JVP verification:

```bash
python experiments/verify_jvp.py --I 60 --N 12 --T 20
```

## Spatial refinement

```bash
python experiments/run_spatial_refinement.py --grids 50 100 200 400 800 1600 --N 80 --T 20
```

The finest grids can be computationally expensive. The script uses continuation in the equilibrium rate path between successive wealth grids.

## Numerical architecture

After monotone spatial discretization, the household problem produces a conservative Metzler generator `A`. The method:

1. collocates the algebraic equilibrium interest rate at interval midpoints;
2. iterates the nonlinear HJB predictor/corrector to a converged midpoint generator `Abar`;
3. uses exactly that same `Abar` in the forward Kolmogorov exponential action;
4. solves interval market-clearing residuals for the stage-rate path;
5. optionally differentiates the converged propagator with Fréchet matrix-exponential actions for matrix-free Newton–Krylov solves.

## Reproducibility note

The `archive/development/` directory preserves intermediate scripts used during method development. The scientifically supported implementation for the manuscript is `src/huggett_exponential_collocation.py` together with `src/huggett_nesvm_core.py`.

## License

No open-source license has been selected yet. Add a license before public archival if redistribution terms are desired.

## Citation

A formal citation and DOI will be added after manuscript submission / archival.
