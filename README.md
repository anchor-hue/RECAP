# RECAP

Run the following commands from this directory. Packaged inputs are in `data/`.
Replace `cuda:0` with `cpu` to run on a CPU.

## Installation

Use Python 3.10 and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

For dataset descriptions, original data sources, and test-mixture construction, see [DATASETS.md](DATASETS.md).
## Benchmark datasets

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

```bash
for design in replicate_holdout_A replicate_holdout_B all_pure_reference; do
  python run_recap.py physical --design "$design" --seed 2021 --device cuda:0
done
python scripts/analyze_physical.py --seed 2021
```

Alternatively, train the three designs in parallel before running the analysis:

```bash
python scripts/run_downstream_models.py --experiment physical --devices cuda:0,cuda:1,cuda:2
```

## SCPro spatial analysis

```bash
for mode in immune9 treg8 main14; do
  python run_recap.py scpro --mode "$mode" --seed 2021 --device cuda:0
done
python scripts/analyze_scpro.py --seed 2021
```

Alternatively, train the three panels in parallel before running the analysis:

```bash
python scripts/run_downstream_models.py --experiment scpro --devices cuda:0,cuda:1,cuda:2
```

## Evaluation tests

```bash
python -m unittest discover -s tests -v
```
