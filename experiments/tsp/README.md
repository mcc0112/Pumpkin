# TSP Benchmark Experiments

This folder runs the same two-propagator comparison as the synthetic
circuit experiments, but against real **TSPLIB** instances.


---

## Step 0 — Download TSPLIB instances

TSPLIB is freely available from the University of Heidelberg:

```
http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/
http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/atsp/
```

Good starter instances (small → medium → large):

| File | n | Type | Notes |
|------|---|------|-------|
| `burma14.tsp`  |  14 | GEO     | tiny, useful for smoke tests |
| `berlin52.tsp` |  52 | EUC_2D  | classic benchmark |
| `eil76.tsp`    |  76 | EUC_2D  | |
| `kroA100.tsp`  | 100 | EUC_2D  | |
| `pr76.tsp`     |  76 | EUC_2D  | |
| `gr17.tsp`     |  17 | EXPLICIT (GEO matrix) | |
| `ftv33.atsp`   |  34 | EXPLICIT FULL_MATRIX  | asymmetric |
| `ft53.atsp`    |  53 | EXPLICIT FULL_MATRIX  | asymmetric |

Download the `.tsp` / `.atsp` files (not the `.opt.tour` files) and drop
them into `experiments/tsp/tsplib/`.

---

## Step 1 — Run baseline propagator

From the **Pumpkin project root**:

```powershell
python experiments\tsp\run_tsp_experiments.py --release --timeout 300
```

This will:
1. Parse every `.tsp` / `.atsp` in `tsplib/`
2. Write a `.mzn` into `instances/<name>/`
3. Flatten to `.fzn` via `minizinc --solver pumpkin -c`
4. Solve with `cargo run -p pumpkin-solver -- file.fzn -s`
5. Save `results/stats_<timestamp>.csv`

---

## Step 2 — Switch branch and re-run

```powershell
git checkout your-new-propagator-branch
cargo build --release

# --no-convert reuses the EXACT same .mzn and .fzn files
python experiments\tsp\run_tsp_experiments.py --release --timeout 300 --no-convert
```

`--no-convert` skips steps 1–3 above so both propagators run against
byte-for-byte identical `.fzn` files.

---

## Step 3 — Compare

```powershell
# Auto-detects the two most-recent stats_*.csv files
python experiments\tsp\analyse_tsp_results.py
```

Or explicitly:

```powershell
python experiments\tsp\analyse_tsp_results.py `
    --baseline experiments\tsp\results\stats_20260513_120000.csv `
    --new      experiments\tsp\results\stats_20260513_160000.csv
```

### Outputs

| File | Description |
|------|-------------|
| `results/comparison_<ts>.csv` | Per-instance, per-metric table with abs/rel diff |
| `results/plot_tsp_runtime.png` | Grouped bar: wall-clock time per instance |
| `results/plot_tsp_search_reduction.png` | Grouped bar: failures per instance |
| `results/plot_tsp_explanation.png` | Grouped bar: noGoods per instance |
| `results/plot_tsp_speedup.png` | Horizontal bar: speedup ratio per instance |

---

## Model choice: why total tour length?

The generated `.mzn` minimises **total tour length** (sum of all legs),
not the longest leg used in the synthetic experiments.  Reasons:

- This is the canonical TSP objective — every TSPLIB instance has a
  known optimal total-cost solution you can verify against.
- "Longest leg" is a non-standard objective that would make results
  incomparable to any published benchmark.
- The circuit constraint (`circuit(succ)`) is identical in both models,
  so propagator behaviour is directly comparable across experiment sets.

---

## Supported EDGE_WEIGHT_TYPE values

`tsp_to_mzn.py` handles:

| Type | Description |
|------|-------------|
| `EUC_2D`  | Euclidean, rounded to nearest int |
| `CEIL_2D` | Euclidean, ceiling |
| `ATT`     | AT&T pseudo-Euclidean |
| `GEO`     | Great-circle (latitude/longitude) |
| `EXPLICIT` | Matrix given directly (FULL_MATRIX, LOWER_DIAG_ROW, UPPER_DIAG_ROW, LOWER_ROW, UPPER_ROW) |

If you encounter an unsupported type the script will exit with a clear
error message.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `No .tsp or .atsp files found` | Check files are in `experiments\tsp\tsplib\` |
| `fzn_missing` in results CSV | You used `--no-convert` before running the baseline; run without it first |
| `Unsupported EDGE_WEIGHT_TYPE` | Check the type in the .tsp header; open an issue or add the type to `tsp_to_mzn.py` |
| Solver times out on all instances | Reduce instance size or increase `--timeout` |