# Structure-Preserving Exponential Collocation for Continuous-Time Heterogeneous-Agent Equilibria

Research code accompanying the manuscript by Braden Holst and Ralph E. White on a stage-centered exponential collocation method for continuous-time Huggett equilibrium transitions.

## Repository contents

- `src/huggett_nesvm_core.py` — cleaned monotone/upwind Huggett discretization, stationary-equilibrium routines, exponential affine steps, and Fréchet actions.
- `src/huggett_exponential_collocation.py` — final shared-stage HJB/Kolmogorov DAE solver. The same converged midpoint Markov generator is used in both backward HJB and forward Kolmogorov propagation.
- `experiments/replicate_achdou_moll.py` — independent Python reproduction of the canonical Achdou–Moll finite-difference Huggett MIT-shock transition benchmark.
- `experiments/run_smooth_ge.py` — smooth general-equilibrium transition.
- `experiments/run_mit_shock.py` — abrupt MIT-shock transition.
- `experiments/run_spatial_refinement.py` — spatial-refinement driver.
- `experiments/verify_jvp.py` — finite-difference verification of the Fréchet/Jacobian-vector product.
- `results/reference/` — reference Achdou–Moll reproduction data.
- `results/paper/` — selected paper benchmark and spatial-refinement tables.
- `results/generated/` — small generated verification data included for reproducibility.
- `validation/` — notes describing the manuscript validation hierarchy.
- `docs/` — documentation notes.

The public repository is intentionally centered on the final reproducible implementation rather than every superseded development prototype.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
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

The finest grids can be computationally expensive. The driver uses continuation in the equilibrium rate path between successive wealth grids.

## Numerical architecture

After monotone spatial discretization, the household problem produces a conservative Metzler generator `A`. The method:

1. collocates the algebraic equilibrium interest rate at interval midpoints;
2. iterates the nonlinear HJB predictor/corrector to a converged midpoint generator `Abar`;
3. uses exactly that same `Abar` in the forward Kolmogorov exponential action;
4. solves interval market-clearing residuals for the stage-rate path; and
5. differentiates the converged propagator with Fréchet matrix-exponential actions for matrix-free Newton–Krylov solves.

## Reproducibility status

The final publication implementation and the principal reproduction/verification drivers are included. The code was syntax checked locally, and the cleaned core/shared-stage implementation was smoke-tested against finite-difference JVPs before upload.

The original Guo lithium-ion P2D MATLAB files and published papers are not redistributed here. They motivated the numerical architecture but are not required to reproduce the economics results.

## License

No open-source license has been selected yet. Add a license before permanent public archival if redistribution terms are desired.

## Citation

A formal citation and DOI will be added after manuscript submission / archival.
