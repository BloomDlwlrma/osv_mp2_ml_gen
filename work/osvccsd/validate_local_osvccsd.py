#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np

from orca_out_parser import iter_pair_corr, molname_from_path


SHARED_DATASETS = [
    "Coulomb(pairlist)",
    "Exchange(pairlist)",
    "Smat_osv(pairlist)",
    "Fmat_osv(pairlist)",
    "Kmat_osv(pairlist)",
    "kmat_osv_dim(pairlist)",
    "loc_fock(nocc,nocc)",
    "mo_list",
    "nocc",
    "nocc_core",
    "nosv(mo_list)",
    "s_ratio(nocc,nocc)",
]


def _load_module(path: Path, module_name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _copy_selected_groups(src: Path, dst: Path, names: list[str]) -> None:
    with h5py.File(src, "r") as fin, h5py.File(dst, "w") as fout:
        for name in names:
            if name not in fin:
                continue
            fin.copy(name, fout)


def _find_orca_out(orca_root: Path, molid: str) -> Path:
    matches = sorted(orca_root.rglob(f"*{molid}*.out"))
    if not matches:
        raise FileNotFoundError(f"no ORCA output found for {molid} under {orca_root}")
    return matches[0]


def _assert_suffix_datasets(raw_fea: Path, molname: str, suffix: str, expected_keys: list[str]) -> None:
    with h5py.File(raw_fea, "r") as h5:
        grp = h5[molname]
        for key in expected_keys:
            name = f"{key}{suffix}"
            if name not in grp:
                raise AssertionError(f"missing dataset: {molname}/{name}")


def _compare_shared_data(src: Path, dst: Path, molname: str) -> None:
    with h5py.File(src, "r") as fin, h5py.File(dst, "r") as fout:
        src_grp = fin[molname]
        dst_grp = fout[molname]
        for key in SHARED_DATASETS:
            if key not in src_grp or key not in dst_grp:
                continue
            src_arr = np.asarray(src_grp[key][()])
            dst_arr = np.asarray(dst_grp[key][()])
            if not np.array_equal(src_arr, dst_arr):
                raise AssertionError(f"shared dataset changed for {molname}: {key}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-fea", required=True, help="Source ml_features.hdf5")
    parser.add_argument("--orca-root", required=True, help="Root directory containing ORCA .out files")
    parser.add_argument("--mols", nargs="+", default=["010295", "009296", "009337"], help="Molecule ids to validate")
    args = parser.parse_args()

    raw_fea = Path(args.raw_fea)
    orca_root = Path(args.orca_root)
    validate_ids = list(dict.fromkeys(args.mols))

    with tempfile.TemporaryDirectory(prefix="osvccsd_validate_") as td:
        td_path = Path(td)
        temp_raw = td_path / "ml_features.hdf5"
        temp_pair = td_path / "pair_energy_osvccsd.hdf5"
        temp_triples = td_path / "triples_pairsum.hdf5"

        selected_groups = [f"dsgdb9nsd_{m}" for m in validate_ids]
        _copy_selected_groups(raw_fea, temp_raw, selected_groups)

        pair_writer = _load_module(Path(__file__).with_name("01_extract_pair_corr_eng.py"), "pair_writer")
        triple_writer = _load_module(Path(__file__).with_name("02_extract_triples_correction.py"), "triple_writer")

        for molid in validate_ids:
            molname = f"dsgdb9nsd_{molid}"
            orca_out = _find_orca_out(orca_root, molid)
            pair_writer._write_hdf5(temp_pair, [orca_out], temp_raw, "osvccsd", update_raw_features=True)

            with h5py.File(temp_pair, "r") as h5:
                grp = h5[molname]
                pair_out = np.asarray(grp["pair_ene"][()])
                pair_ids = np.asarray(grp["pairlist"][()])
                expected_pairs = len(list(iter_pair_corr(orca_out)))
                print(f"[pair] {molname}: kept={len(pair_out)} expected_orca_pairs={expected_pairs} skipped={grp.attrs['n_skipped_orca_pairs']}")
                if len(pair_out) != expected_pairs:
                    raise AssertionError(f"pair count mismatch for {molname}")

            _assert_suffix_datasets(temp_raw, molname, "_ccsd", ["pairlist", "pair_ene", "pairlist_offdiag"])
            _compare_shared_data(raw_fea, temp_raw, molname)

            pair_writer._write_hdf5(temp_pair, [orca_out], temp_raw, "osvccsdt", update_raw_features=True)
            with h5py.File(temp_pair, "r") as h5:
                grp = h5[molname]
                print(f"[pair+triple] {molname}: kept={len(grp['pair_ene'])} n_triple_pairs={grp.attrs['n_triple_pairs']}")
            _assert_suffix_datasets(temp_raw, molname, "_ccsdt", ["pairlist", "pair_ene", "pairlist_offdiag"])

            triple_writer._write_hdf5(temp_triples, [orca_out], True, temp_raw, update_raw_features=True)
            with h5py.File(temp_triples, "r") as h5:
                grp = h5[molname]
                print(f"[triples] {molname}: n_pairs={grp.attrs['n_pairs']}")
            _assert_suffix_datasets(temp_raw, molname, "_ccsd", ["triple_pairlist", "triple_pair_ene"])

        print("validation_ok True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
