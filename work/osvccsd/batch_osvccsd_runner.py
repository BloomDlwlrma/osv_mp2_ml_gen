#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import tempfile
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from orca_out_parser import _expand_inputs, molname_from_path
import h5py
from typing import Set


def _load_pair_writer():
    script_path = SCRIPT_DIR / "01_extract_pair_corr_eng.py"
    spec = importlib.util.spec_from_file_location("osvccsd_pair_writer", script_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("failed to load pair writer")
    spec.loader.exec_module(module)
    return module


def _load_collect_hdf5():
    collect_path = Path(__file__).resolve().parents[1] / "collect_hdf5.py"
    spec = importlib.util.spec_from_file_location("collect_hdf5_mod", collect_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("failed to load collect_hdf5")
    spec.loader.exec_module(module)
    return module.collect_hdf5


def _write_chunks(input_paths: list[Path], raw_fea: Path | None, qm_method: str, chunk_size: int, tmp_dir: Path) -> list[Path]:
    pair_writer = _load_pair_writer()
    chunk_files: list[Path] = []

    for idx in range(0, len(input_paths), chunk_size):
        chunk = input_paths[idx : idx + chunk_size]
        chunk_out = tmp_dir / f"chunk_{idx // chunk_size:05d}.h5"
        pair_writer._write_hdf5(chunk_out, chunk, raw_fea, qm_method)
        chunk_files.append(chunk_out)

    return chunk_files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="ORCA .out files")
    parser.add_argument("--output", required=True, help="Final HDF5 output path")
    parser.add_argument("--raw-fea", default=None, help="ml_features.hdf5 for pair ordering")
    parser.add_argument("--qm-method", choices=["osvccsd", "osvccsdt"], default="osvccsdt")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--merge-mode", choices=["OVERWRITE", "APPEND_MISSING"], default="OVERWRITE")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without writing HDF5")
    args = parser.parse_args()

    input_paths = _expand_inputs(args.inputs)
    if not input_paths:
        raise ValueError("no input files")

    out_path = Path(args.output)
    out_dir = out_path.parent
    raw_fea = Path(args.raw_fea) if args.raw_fea else None

    # Determine target molecules: if both pair_energy.hdf5 and ml_features.hdf5
    # already exist in the output directory, use the groups from pair_energy.hdf5
    # as the processing set. Otherwise process all input .out files.
    pair_h5 = out_dir / "pair_energy.hdf5"
    ml_h5 = out_dir / "ml_features.hdf5"
    target_mols: Set[str] | None = None
    if pair_h5.exists() and ml_h5.exists():
        try:
            with h5py.File(pair_h5, "r") as hfin:
                target_mols = set(hfin.keys())
        except Exception:
            target_mols = None

    # Map inputs by molname and filter if target_mols specified
    has_missing = False
    if target_mols is not None:
        mol_map = {}
        for p in input_paths:
            mol_map.setdefault(molname_from_path(p), p)
        missing = sorted([m for m in target_mols if m not in mol_map])
        if missing:
            has_missing = True
            print(f"[batch_osvccsd_runner] Warning: {len(missing)} mols in '{pair_h5}' missing .out files: {missing[:5]}{('...' if len(missing)>5 else '')}")
        filtered = [mol_map[m] for m in sorted(target_mols) if m in mol_map]
        input_paths = filtered

    if has_missing:
        print("[batch_osvccsd_runner] Aborting: some molecules have missing .out files. Returning non-zero exit code.")
        return 1

    if not input_paths:
        print("[batch_osvccsd_runner] No matched input .out files to process after filtering.")
        return 0

    # Prepare output filenames for the two QM variants
    out_osvccsd = out_dir / "pair_energy_osvccsd.hdf5"
    out_osvccsdt = out_dir / "pair_energy_osvccsdt.hdf5"

    collect_hdf5 = _load_collect_hdf5()

    # Helper to run write+collect for a given qm_method and output path
    def _produce(output_path: Path, qm_method: str) -> None:
        if len(input_paths) <= args.chunk_size:
            pair_writer = _load_pair_writer()
            pair_writer._write_hdf5(output_path, input_paths, raw_fea, qm_method)
            return

        with tempfile.TemporaryDirectory(prefix=f"osvccsd_{qm_method}_") as tmp_root:
            tmp_dir = Path(tmp_root)
            chunk_files = _write_chunks(input_paths, raw_fea, qm_method, args.chunk_size, tmp_dir)
            collect_hdf5([str(p) for p in chunk_files], str(output_path), mode=args.merge_mode)

    # Processing order: first osvccsd (pair-only), then osvccsdt (triples added)
    print(f"[batch_osvccsd_runner] Processing {len(input_paths)} .out files -> {out_osvccsd.name} (osvccsd)")
    print(f"[batch_osvccsd_runner] Then {len(input_paths)} .out files -> {out_osvccsdt.name} (osvccsdt)")
    print(f"[batch_osvccsd_runner] Molecules to process: {sorted([molname_from_path(p) for p in input_paths])}")
    
    if args.dry_run:
        print(f"[batch_osvccsd_runner] **DRY-RUN MODE** - No HDF5 files will be written. Exiting.")
        return 0

    _produce(out_osvccsd, "osvccsd")

    print(f"[batch_osvccsd_runner] Processing {len(input_paths)} .out files -> {out_osvccsdt.name} (osvccsdt)")
    _produce(out_osvccsdt, "osvccsdt")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
