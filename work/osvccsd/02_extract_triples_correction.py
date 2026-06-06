#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from ml_features_local import load_allowed_molecules, write_method_datasets
from orca_out_parser import _expand_inputs, iter_triples, molname_from_path


def _write_hdf5(
    output_path: Path,
    source_files: list[Path],
    pair_sum: bool,
    raw_fea_path: Path | None = None,
    update_raw_features: bool = True,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    allowed_mols = load_allowed_molecules(raw_fea_path) if raw_fea_path is not None else set()

    with h5py.File(output_path, "w") as hout:
        hout.attrs["source_method"] = "osvccsd"
        hout.attrs["pair_sum_triples"] = bool(pair_sum)
        if raw_fea_path is not None:
            hout.attrs["raw_features_h5"] = str(raw_fea_path)

        for path in source_files:
            molname = molname_from_path(path)
            if allowed_mols and molname not in allowed_mols:
                print(f"[osvccsd] Skipping {molname}: not present in ml_features.hdf5")
                continue

            grp = hout.create_group(molname)
            if pair_sum:
                acc: dict[tuple[int, int], float] = defaultdict(float)
                for record in iter_triples(path):
                    acc[(record.i, record.j)] += record.et_ijk
                pairs = sorted(acc.items())
                pairlist = np.asarray([[i, j] for (i, j), _ in pairs], dtype=np.int32) if pairs else np.empty((0, 2), dtype=np.int32)
                et = np.fromiter((value for _, value in pairs), dtype=np.float64, count=len(pairs)) if pairs else np.empty((0,), dtype=np.float64)
                grp.create_dataset("pairlist", data=pairlist)
                grp.create_dataset("pair_ene", data=et)
                grp.attrs["n_pairs"] = int(len(pairs))
                grp.attrs["mode"] = "pair_sum"

                # embed occupancy and screening metadata from raw features when available
                if raw_fea_path is not None:
                    try:
                        with h5py.File(raw_fea_path, "r") as ffea:
                            if molname in ffea:
                                grp_in_fea = ffea[molname]
                                for key in ["nocc", "pairlist_screened"]:
                                    if key in grp_in_fea and key not in grp:
                                        grp.create_dataset(key, data=np.asarray(grp_in_fea[key][:]))
                    except Exception:
                        # don't fail the whole extraction if embedding metadata is not possible
                        pass

                if raw_fea_path is not None and update_raw_features:
                    write_method_datasets(
                        raw_fea_path,
                        molname,
                        "osvccsd",
                        {
                            "triple_pairlist": pairlist,
                            "triple_pair_ene": et,
                        },
                    )
            else:
                triples = list(iter_triples(path))
                triple_idx = np.asarray([[r.i, r.j, r.k] for r in triples], dtype=np.int32) if triples else np.empty((0, 3), dtype=np.int32)
                triple_ene = np.asarray([r.et_ijk for r in triples], dtype=np.float64) if triples else np.empty((0,), dtype=np.float64)
                grp.create_dataset("triple_idx", data=triple_idx)
                grp.create_dataset("triple_ene", data=triple_ene)
                grp.attrs["n_triples"] = int(len(triples))
                grp.attrs["mode"] = "triple_list"

                # embed occupancy and screening metadata from raw features when available
                if raw_fea_path is not None:
                    try:
                        with h5py.File(raw_fea_path, "r") as ffea:
                            if molname in ffea:
                                grp_in_fea = ffea[molname]
                                for key in ["nocc", "pairlist_screened"]:
                                    if key in grp_in_fea and key not in grp:
                                        grp.create_dataset(key, data=np.asarray(grp_in_fea[key][:]))
                    except Exception:
                        pass

                if raw_fea_path is not None and update_raw_features:
                    write_method_datasets(
                        raw_fea_path,
                        molname,
                        "osvccsd",
                        {
                            "triple_idx": triple_idx,
                            "triple_ene": triple_ene,
                        },
                    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="ORCA .out files")
    parser.add_argument("--output", required=True, help="Output HDF5 path")
    parser.add_argument("--pair-sum", action="store_true", help="Store pair-summed triples")
    parser.add_argument("--raw-fea", default=None, help="ml_features.hdf5 for allowlist and in-group dataset updates")
    parser.add_argument("--no-update-raw-fea", action="store_true", help="Do not append method datasets back to ml_features.hdf5")
    args = parser.parse_args()

    input_paths = _expand_inputs(args.inputs)
    if not input_paths:
        raise ValueError("no input files")

    _write_hdf5(
        Path(args.output),
        input_paths,
        args.pair_sum,
        Path(args.raw_fea) if args.raw_fea else None,
        update_raw_features=not args.no_update_raw_fea,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
