#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from orca_out_parser import _expand_inputs, iter_triples, molname_from_path


def _write_hdf5(output_path: Path, source_files: list[Path], pair_sum: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as hout:
        hout.attrs["source_method"] = "osvccsd"
        hout.attrs["pair_sum_triples"] = bool(pair_sum)
        for path in source_files:
            molname = molname_from_path(path)
            grp = hout.create_group(molname)
            if pair_sum:
                acc: dict[tuple[int, int], float] = defaultdict(float)
                for record in iter_triples(path):
                    acc[(record.i, record.j)] += record.et_ijk
                if acc:
                    pairs = sorted(acc.items())
                    pairlist = np.asarray([[i, j] for (i, j), _ in pairs], dtype=np.int32)
                    et = np.fromiter((value for _, value in pairs), dtype=np.float64, count=len(pairs))
                    grp.create_dataset("pairlist", data=pairlist)
                    grp.create_dataset("pair_ene", data=et)
            else:
                triples = list(iter_triples(path))
                if triples:
                    triple_idx = np.asarray([[r.i, r.j, r.k] for r in triples], dtype=np.int32)
                    triple_ene = np.asarray([r.et_ijk for r in triples], dtype=np.float64)
                    grp.create_dataset("triple_idx", data=triple_idx)
                    grp.create_dataset("triple_ene", data=triple_ene)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="ORCA .out files")
    parser.add_argument("--output", required=True, help="Output HDF5 path")
    parser.add_argument("--pair-sum", action="store_true", help="Store pair-summed triples")
    args = parser.parse_args()

    input_paths = _expand_inputs(args.inputs)
    if not input_paths:
        raise ValueError("no input files")

    _write_hdf5(Path(args.output), input_paths, args.pair_sum)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
