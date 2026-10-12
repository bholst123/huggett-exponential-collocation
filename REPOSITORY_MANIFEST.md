# Repository manifest

## Publication implementation
- `src/huggett_nesvm_core.py`
- `src/huggett_exponential_collocation.py`

## Reproduction drivers
- `experiments/replicate_achdou_moll.py`
- `experiments/run_smooth_ge.py`
- `experiments/run_mit_shock.py`
- `experiments/run_spatial_refinement.py`
- `experiments/verify_jvp.py`

## Development archive
Earlier prototypes and intermediate research scripts are retained locally for provenance and may be added under `archive/development/` as needed.

## Data / selected outputs
Reference and selected paper result CSVs and figures are included under `results/`.

## Excluded on purpose
The original Guo lithium-ion P2D MATLAB source/workspace and published PDFs are not redistributed in this economics-code repository. They motivated the numerical architecture but are not required to reproduce the Huggett results. This avoids mixing third-party source/data with the code archive.
