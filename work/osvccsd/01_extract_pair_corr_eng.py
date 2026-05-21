#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

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


def _write_hdf5(output_path: Path, source_files: list[Path], raw_fea_path: Path | None, qm_method: str) -> None:
    pair_map: dict[str, dict[tuple[int, int], float]] = defaultdict(dict)
    triple_count_map: dict[str, int] = defaultdict(int)
    ignored_pair_count: dict[str, int] = defaultdict(int)
    for path in source_files:
        molname = molname_from_path(path)
        pair_order = None
        nocc = None
        allowed_pairs: set[int] | None = None
        if raw_fea_path is not None:
            loaded = _load_pair_metadata(raw_fea_path, molname)
            if loaded is not None:
                pair_order, nocc, allowed_pairs, screened_pairs = loaded
        for record in iter_pair_corr(path):
            if allowed_pairs is not None and nocc is not None:
                # ml_features stores raw ORCA pairs (possibly i > j) as j_raw*nocc+i_raw
                # But iter_pair_corr normalizes to i <= j, so encode as i_norm*nocc+j_norm
                pair_id = record.i * nocc + record.j
                if pair_id not in allowed_pairs:
                    ignored_pair_count[molname] += 1
                    continue
            pair_map[molname][(record.i, record.j)] = record.ep_final
        if qm_method == "osvccsdt":
            triple_map = _aggregate_triples(path)
            triple_count_map[molname] = len(triple_map)
            for pair, et_ijk in triple_map.items():
                if allowed_pairs is not None and nocc is not None:
                    # pair is already normalized (i <= j) from iter_triples
                    pair_id = pair[0] * nocc + pair[1]
                    if pair_id not in allowed_pairs:
                        continue
                if pair in pair_map[molname]:
                    pair_map[molname][pair] += et_ijk
                else:
                    print(f"  WARNING: triple pair {pair} has no corresponding pair_corr for {molname}")
                    pair_map[molname][pair] = et_ijk

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as hout:
        hout.attrs["source_method"] = qm_method
        if raw_fea_path is not None:
            hout.attrs["raw_features_h5"] = str(raw_fea_path)

        for molname, records in pair_map.items():
            grp = hout.create_group(molname)
            pair_order = None
            nocc = None
            screened_pairs: set[int] = set()
            ignored_count = ignored_pair_count.get(molname, 0)
            if raw_fea_path is not None:
                loaded = _load_pair_metadata(raw_fea_path, molname)
                if loaded is not None:
                    pair_order, nocc, allowed_pairs, screened_pairs = loaded
                    grp.attrs["n_screened_remote_pairs"] = len(screened_pairs)
            if pair_order is not None and nocc is not None:
                pair_ene = np.empty(len(pair_order), dtype=np.float64)
                missing_pairs = []
                for idx, pair_id in enumerate(pair_order):
                    # pair_order contains pair IDs from ml_features (may be raw ORCA pairs: j_raw*nocc+i_raw)
                    # iter_pair_corr returns normalized pairs (i <= j), so try both orderings
                    i_norm = int(pair_id // nocc)
                    j_norm = int(pair_id % nocc)
                    value = records.get((i_norm, j_norm))
                    if value is None:
                        value = records.get((j_norm, i_norm))
                    if value is None:
                        missing_pairs.append((i_norm, j_norm))
                        value = np.nan
                    pair_ene[idx] = value
                if missing_pairs or ignored_count:
                    print(f"  INFO [{molname}]: ignored_orca_pairs={ignored_count}, missing_kept_pairs={len(missing_pairs)}")
                    if missing_pairs:
                        print(f"    first_missing_kept_pairs={missing_pairs[:5]}")
            else:
                pairs = sorted(records.items())
                pair_ene = np.fromiter((value for _, value in pairs), dtype=np.float64, count=len(pairs))
            grp.create_dataset("pair_ene", data=pair_ene)
            if raw_fea_path is not None:
                grp.attrs["n_ignored_orca_pairs"] = ignored_count
            if qm_method == "osvccsdt":
                grp.attrs["n_triple_pairs"] = triple_count_map.get(molname, 0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="ORCA .out files")
    parser.add_argument("--output", required=True, help="Output HDF5 path")
    parser.add_argument("--raw-fea", default=None, help="ml_features.hdf5 for pair ordering")
    parser.add_argument(
        "--qm-method",
        choices=["osvccsd", "osvccsdt"],
        default="osvccsd",
        help="`osvccsd` writes pair-only energies; `osvccsdt` adds triples corrections to pairs.",
    )
    args = parser.parse_args()

    input_paths = _expand_inputs(args.inputs)
    if not input_paths:
        raise ValueError("no input files")

    _write_hdf5(
        Path(args.output),
        input_paths,
        Path(args.raw_fea) if args.raw_fea else None,
        args.qm_method,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
