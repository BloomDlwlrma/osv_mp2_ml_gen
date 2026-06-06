#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from ml_features_local import load_allowed_molecules, write_method_datasets
from orca_out_parser import _expand_inputs, iter_pair_corr, iter_triples, molname_from_path


def _load_pair_metadata(raw_fea_path: Path, molname: str) -> tuple[np.ndarray, int, set[int], set[int]] | None:
    if not raw_fea_path.exists():
        return None
    with h5py.File(raw_fea_path, "r") as h5:
        if molname not in h5:
            return None
        grp = h5[molname]
        nocc = int(np.asarray(grp["nocc"])[0])
        pairlist = np.asarray(grp["pairlist"])
        screened = np.asarray(grp["pairlist_screened"]) if "pairlist_screened" in grp else np.asarray([])
        if pairlist.ndim == 2:
            pair_idx = pairlist[:, 0].astype(np.int64) * nocc + pairlist[:, 1].astype(np.int64)
        else:
            pair_idx = pairlist.astype(np.int64)
        allowed_pairs = {int(x) for x in pair_idx.reshape(-1)}
        screened_pairs = {int(x) for x in screened.reshape(-1)}
        return pair_idx, nocc, allowed_pairs, screened_pairs


def _aggregate_triples(path: Path) -> dict[tuple[int, int], float]:
    acc: dict[tuple[int, int], float] = defaultdict(float)
    for record in iter_triples(path):
        acc[(record.i, record.j)] += record.et_ijk
    return acc


def _pairid_to_pair(pair_id: int, nocc: int) -> tuple[int, int]:
    return int(pair_id // nocc), int(pair_id % nocc)


def _write_hdf5(
    output_path: Path,
    source_files: list[Path],
    raw_fea_path: Path | None,
    qm_method: str,
    update_raw_features: bool = True,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    allowed_mols = load_allowed_molecules(raw_fea_path) if raw_fea_path is not None else set()

    with h5py.File(output_path, "w") as hout:
        hout.attrs["source_method"] = qm_method
        hout.attrs["pair_output_mode"] = "orca_kept_pairs_only"
        if raw_fea_path is not None:
            hout.attrs["raw_features_h5"] = str(raw_fea_path)

        for path in source_files:
            molname = molname_from_path(path)
            if allowed_mols and molname not in allowed_mols:
                print(f"[osvccsd] Skipping {molname}: not present in ml_features.hdf5")
                continue

            loaded = _load_pair_metadata(raw_fea_path, molname) if raw_fea_path is not None else None
            pair_order: np.ndarray | None = None
            nocc: int | None = None
            screened_pairs: set[int] | None = None
            if loaded is not None: 
                pair_order, nocc, _, screened_pairs = loaded

            pair_records: dict[tuple[int, int], float] = {}
            for record in iter_pair_corr(path):
                pair_records[(record.i, record.j)] = record.ep_final

            triple_map: dict[tuple[int, int], float] = {}
            if qm_method == "osvccsdt":
                triple_map = _aggregate_triples(path)

            if pair_order is not None and nocc is not None:
                kept_pair_ids: list[int] = []
                kept_pair_ene: list[float] = []
                skipped_pair_count = 0
                for pair_id in pair_order:
                    i_norm, j_norm = _pairid_to_pair(int(pair_id), nocc)
                    value = pair_records.get((i_norm, j_norm))
                    if value is None:
                        value = pair_records.get((j_norm, i_norm))
                    if value is None:
                        skipped_pair_count += 1
                        continue
                    if qm_method == "osvccsdt":
                        triple_value = triple_map.get((i_norm, j_norm))
                        if triple_value is None:
                            triple_value = triple_map.get((j_norm, i_norm), 0.0)
                        value += triple_value
                    kept_pair_ids.append(int(pair_id))
                    kept_pair_ene.append(float(value))

                pairlist_out = np.asarray(kept_pair_ids, dtype=np.int64)
                pair_ene_out = np.asarray(kept_pair_ene, dtype=np.float64)
                offdiag_out = np.asarray(
                    [pid for pid in kept_pair_ids if _pairid_to_pair(pid, nocc)[0] != _pairid_to_pair(pid, nocc)[1]],
                    dtype=np.int64,
                )
                suffix_payload = {
                    "pairlist": pairlist_out,
                    "pair_ene": pair_ene_out,
                    "pairlist_offdiag": offdiag_out,
                }
            else:
                raise RuntimeError(
                    f"[{molname}] missing pair metadata from --raw-fea; "
                    "pairlist-driven CCSD/CCSDT extraction requires 1D pairlist + nocc."
                )

            grp = hout.create_group(molname)
            grp.create_dataset("pairlist", data=pairlist_out)
            grp.create_dataset("pair_ene", data=pair_ene_out)
            # embed occupancy and screening metadata when available so pair_energy files are self-contained
            if nocc is not None:
                # store nocc as a 1-element integer dataset (keeps compatibility with existing readers)
                grp.create_dataset("nocc", data=np.asarray([int(nocc)], dtype=np.int32))
            if screened_pairs:
                # store screened pair indices as an integer array
                grp.create_dataset("pairlist_screened", data=np.asarray(sorted(list(screened_pairs)), dtype=np.int64))
            grp.attrs["n_kept_pairs"] = int(len(pair_ene_out))
            grp.attrs["n_skipped_orca_pairs"] = int(skipped_pair_count)
            grp.attrs["method"] = qm_method

            if qm_method == "osvccsdt":
                grp.attrs["n_triple_pairs"] = int(len(triple_map))

            if raw_fea_path is not None and update_raw_features:
                write_method_datasets(raw_fea_path, molname, qm_method, suffix_payload)

            print(
                f"  INFO [{molname}]: skipped_orca_pairs={skipped_pair_count}, "
                f"kept_pairs={len(pair_ene_out)}, missing_kept_pairs=0"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="ORCA .out files")
    parser.add_argument("--output", required=True, help="Output HDF5 path")
    parser.add_argument("--raw-fea", required=True, help="ml_features.hdf5 for pair ordering and update target")
    parser.add_argument(
        "--qm-method",
        choices=["osvccsd", "osvccsdt"],
        default="osvccsd",
        help="`osvccsd` writes pair-only energies; `osvccsdt` adds triples corrections to pairs.",
    )
    parser.add_argument("--no-update-raw-fea", action="store_true", help="Do not append method datasets back to ml_features.hdf5")
    args = parser.parse_args()

    input_paths = _expand_inputs(args.inputs)
    if not input_paths:
        raise ValueError("no input files")

    _write_hdf5(
        Path(args.output),
        input_paths,
        Path(args.raw_fea),
        args.qm_method,
        update_raw_features=not args.no_update_raw_fea,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
