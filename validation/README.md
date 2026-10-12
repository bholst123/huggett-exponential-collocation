# Validation scripts

The manuscript reports several separate validation layers:

- smooth prescribed-path temporal convergence against an independent BDF method-of-lines reference;
- comparison with first-order semi-implicit and second-order Crank-Nicolson schemes;
- positivity / stochastic-semigroup diagnostics;
- event-alignment tests for known discontinuities;
- stage-collocation / checkerboard diagnostics;
- full general-equilibrium temporal and spatial refinement;
- Fréchet/JVP finite-difference verification.

The recommended publication solver is the cleaned shared-stage code in `src/`.
