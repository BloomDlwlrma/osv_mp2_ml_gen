# OSVCCSD Current Workflow (Detailed)

This document describes the **current** OSVCCSD data workflow implemented in this folder, focusing on:

- Input/output contracts
- Execution order
- How pair energies and triples corrections are written
- How this workflow integrates with `ml_features.hdf5`
- Validation and troubleshooting

---

## 1. Overview

The workflow turns ORCA output files (`*.out`) into HDF5 datasets for downstream ML training.

Current main outputs are:

- `pair_energy_osvccsd.hdf5` (pair-only correlation energies)
- `pair_energy_osvccsdt.hdf5` (pair energies + triples corrections)

Main orchestrator:

- `batch_osvccsd_runner.py`

Core extraction modules:

- `01_extract_pair_corr_eng.py`
- `02_extract_triples_correction.py`
- `orca_out_parser.py`
- `ml_features_local.py`

Validation helper:

- `validate_local_osvccsd.py`

---

## 2. Data Contract

### 2.1 Required Inputs

1. ORCA `.out` files (one per molecule)
2. `ml_features.hdf5` (`--raw-fea`)

`ml_features.hdf5` is not only a reference; it is used to:

- restrict allowed molecules (`load_allowed_molecules`)
- provide pair ordering (`pairlist`, `nocc`, optional `pairlist_screened`)
- optionally store method-specific suffixed datasets back into `ml_features.hdf5`

### 2.2 Expected ORCA naming behavior

Molecule names are inferred by `molname_from_path()` from filename stems, typically like:

- `dsgdb9nsd_000123_...out` -> `dsgdb9nsd_000123`

If naming deviates, mapping may fail and molecule filtering may silently drop entries.

### 2.3 Output naming (batch mode)

`batch_osvccsd_runner.py` writes two fixed files in output directory:

- `pair_energy_osvccsd.hdf5`
- `pair_energy_osvccsdt.hdf5`

Even if `--qm-method` is passed, batch mode currently runs both variants in sequence.

---

## 3. Execution Flow

## Step A. Expand and normalize input file list

Implemented by:

- `_expand_inputs()` in `orca_out_parser.py`

Behavior:

- accepts paths and globs
- deduplicates while preserving first occurrence

## Step B. Filter to molecules present in `ml_features.hdf5`

Implemented by:

- `load_allowed_molecules()` in `ml_features_local.py`
- pre-filter block in `batch_osvccsd_runner.py`

Behavior:

- molecules not present in `ml_features.hdf5` are skipped before extraction

## Step C. Chunking for large jobs

Implemented by:

- `_write_chunks()` and `_produce()` in `batch_osvccsd_runner.py`

Behavior:

- if input count > `--chunk-size`, each chunk is written to temp HDF5
- temp chunk files are merged by `collect_h5.py`

## Step D. Parse ORCA and write pair-energy datasets

Implemented by:

- `_write_hdf5(..., qm_method)` in `01_extract_pair_corr_eng.py`
- parsers `iter_pair_corr()` and `iter_triples()` in `orca_out_parser.py`

For `osvccsd`:

- writes pair correlation energies only

For `osvccsdt`:

- aggregates triples contributions per `(i, j)`
- adds triples contribution to pair energy

Important alignment rule:

- output pair order follows `pairlist` from `ml_features.hdf5`
- ORCA-missing pairs are skipped (not zero-filled)
- skipping stats are recorded in group attrs

## Step E. Optional updates back to `ml_features.hdf5`

Implemented by:

- `write_method_datasets()` in `ml_features_local.py`

Behavior:

- writes suffixed datasets (`_ccsd`, `_ccsdt`) under each molecule group
- skips writing suffixed dataset when equal to base dataset (if enabled)

---

## 4. What each script does

## `orca_out_parser.py`

- Regex parser for:
  - pair correlation section
  - triples lines
- Provides:
  - `iter_pair_corr(path)`
  - `iter_triples(path)`
  - `molname_from_path(path)`

## `01_extract_pair_corr_eng.py`

- Main pair-energy writer
- Reads pair order and occupancy metadata from `--raw-fea`
- Writes per-molecule group with:
  - `pairlist`
  - `pair_ene`
  - `nocc` (embedded)
  - `pairlist_screened` (if available)
  - attrs like kept/skipped counts
- Supports modes:
  - `--qm-method osvccsd`
  - `--qm-method osvccsdt`

## `02_extract_triples_correction.py`

- Triples extractor utility
- Supports two outputs:
  - pair-summed triples (`--pair-sum`)
  - full triple list
- Can also write suffixed triples datasets to `ml_features.hdf5`

## `batch_osvccsd_runner.py`

- End-to-end batch orchestrator
- Handles:
  - input expansion/filtering
  - chunking
  - merge of chunks
  - sequential generation of both final files

## `ml_features_local.py`

- Molecule allowlist loading
- Dataset suffix naming utility
- Safe write helper into `ml_features.hdf5`

## `validate_local_osvccsd.py`

- Local consistency checker for selected molecules
- Verifies:
  - expected pair counts
  - suffixed datasets presence
  - shared feature datasets unchanged

---

## 5. Typical Commands

## 5.1 Batch production (recommended)

```bash
python batch_osvccsd_runner.py \
  "/path/to/orca_out/*.out" \
  --raw-fea /path/to/ml_features.hdf5 \
  --output /path/to/work/pair_energy_osvccsdt.hdf5 \
  --chunk-size 128 \
  --merge-mode OVERWRITE
```

Notes:

- `--output` is used as base directory anchor; batch writes both canonical filenames.
- `--raw-fea` must exist.

## 5.2 Pair-only direct writer

```bash
python 01_extract_pair_corr_eng.py \
  "/path/to/orca_out/*.out" \
  --output /path/to/pair_energy_osvccsd.hdf5 \
  --raw-fea /path/to/ml_features.hdf5 \
  --qm-method osvccsd
```

## 5.3 Pair + triples direct writer

```bash
python 01_extract_pair_corr_eng.py \
  "/path/to/orca_out/*.out" \
  --output /path/to/pair_energy_osvccsdt.hdf5 \
  --raw-fea /path/to/ml_features.hdf5 \
  --qm-method osvccsdt
```

## 5.4 Validation run

```bash
python validate_local_osvccsd.py \
  --raw-fea /path/to/ml_features.hdf5 \
  --orca-root /path/to/orca_out \
  --mols 010295 009296 009337
```

---

## 6. HDF5 Layout (Current)

Per molecule group in pair-energy output:

- `pairlist` : 1D encoded pair IDs (aligned to `ml_features` ordering)
- `pair_ene` : final energies per kept pair
- `nocc` : occupancy count
- `pairlist_screened` : optional screened pair IDs
- attrs:
  - `method`
  - `n_kept_pairs`
  - `n_skipped_orca_pairs`
  - `n_triple_pairs` (for `osvccsdt`)

Batch-level attrs (file root):

- `source_method`
- `pair_output_mode`
- `raw_features_h5`

---

## 7. Failure Modes and Fixes

## Symptom: "No matched input .out files to process"

Likely causes:

- input glob mismatch
- molecule names in ORCA filenames do not map to `ml_features.hdf5` keys

Check:

- ORCA filename stem format
- keys in `ml_features.hdf5`

## Symptom: RuntimeError about missing pair metadata / 1D pairlist

Cause:

- `--raw-fea` molecule group missing required fields (`pairlist`, `nocc`)

Fix:

- regenerate or normalize `ml_features.hdf5` so pairlist-driven extraction can align properly

## Symptom: unexpectedly many skipped pairs

Cause:

- ORCA output missing pair records relative to expected pairlist

Check:

- `n_skipped_orca_pairs` attrs
- parser coverage in `orca_out_parser.py`

## Symptom: final file missing one method output

Cause:

- run interrupted between two `_produce()` calls in batch

Fix:

- rerun batch; ensure output directory writable and enough temporary space

---

## 8. Practical Recommendations

1. Always run with `--raw-fea` from the same dataset family used for training.
2. Keep ORCA output naming stable (`dsgdb9nsd_XXXXXX...`) to avoid molecule mapping issues.
3. For large batches, use chunking (`--chunk-size`) and monitor temporary storage.
4. Run `validate_local_osvccsd.py` on a small molecule set before full production.
5. Treat `pair_energy_osvccsd.hdf5` and `pair_energy_osvccsdt.hdf5` as canonical batch outputs for downstream REPT-dNN pipeline.

---

## 9. Relationship to REPT-dNN downstream

These outputs are consumed by downstream descriptor alignment / training scripts in REPT-dNN pipeline. Current contract is:

- pair energy files are molecule-keyed and pairlist-aligned
- method-specific behavior is encoded in filename and group attrs
- missing ORCA pairs are dropped, not padded

This contract is what downstream merge/validation and training wrappers assume.
