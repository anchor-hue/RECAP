# RECAP

Run the following commands from this directory. Packaged inputs are in `data/`.
Replace `cuda:0` with `cpu` to run on a CPU.

## Installation

Use Python 3.10 and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Benchmark datasets

For dataset descriptions, original data sources, and test-mixture construction, see [DATASETS.md](DATASETS.md).

Run one dataset (dataset names are listed in `data/benchmark_manifest.json`):

```bash
python run_recap.py benchmark --dataset murine --seed 2021 --device cuda:0
```

Run all seven datasets, then evaluate:

```bash
python scripts/run_benchmarks.py --seeds 2021 --devices cuda:0
python scripts/summarize_benchmarks.py --seed 2021
```

To run all ten seeds:

```bash
benchmark_seeds=2021,2022,2023,2024,2025,2026,2027,2028,2029,2030
python scripts/run_benchmarks.py --seeds "$benchmark_seeds" --devices cuda:0
python scripts/summarize_benchmarks.py --seeds "$benchmark_seeds"
```

For multiple GPUs, use `--devices cuda:0,cuda:1`.

## Physical ProteoMixture analysis

The packaged `all_pure_reference` design uses all six pure profiles (two per
component) as the reference and ten measured mixtures as targets. Reference
and target inputs retain the single protein-wise z-score fitted to the pure
reference during preprocessing.

Run the model and summarize fraction recovery without generating figures:

```bash
python scripts/run_proteomixture.py --device cuda:0
python scripts/analyze_proteomixture.py
```

Model outputs are written to `results/physical/all_pure_reference/seed_2021/`;
fraction summaries are written to `results/physical/analysis_output_fraction_only/`.
The predefined mixture proportions are the fraction-recovery endpoint; the
measured mixtures have no independent cell-type-resolved expression truth.

## SCPro spatial analysis

SCPro uses separate scripts and retains its existing model normalization and
training settings. Run the `immune9`, `treg8`, and `main14` modes, then compute
the spatial summaries and T8 gene-set analysis. These scripts do not draw figures.

```bash
python scripts/run_scpro.py --devices cuda:0
python scripts/analyze_scpro.py --seed 2021
```

For multiple GPUs, use `--devices cuda:0,cuda:1,cuda:2`.
SCPro model outputs are under `results/scpro/`; summaries are under
`analysis/scpro/`.

## Evaluation tests

```bash
python -m unittest discover -s tests -v
```
