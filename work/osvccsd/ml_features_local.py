from __future__ import annotations

from pathlib import Path
from typing import Mapping

import h5py
import numpy as np


def method_suffix(qm_method: str) -> str:
    if qm_method == "osvccsd":
        return "_ccsd"
    if qm_method == "osvccsdt":
        return "_ccsdt"
    raise ValueError(f"Unsupported qm_method: {qm_method}")


def load_allowed_molecules(raw_fea_path: Path) -> set[str]:
    if not raw_fea_path.exists():
        return set()
    with h5py.File(raw_fea_path, "r") as h5:
        return set(h5.keys())


def _as_array(value) -> np.ndarray:
    return np.asarray(value)


def write_dataset(grp: h5py.Group, name: str, data) -> None:
    if name in grp:
        del grp[name]
    grp.create_dataset(name, data=_as_array(data))


def write_method_datasets(
    raw_fea_path: Path,
    molname: str,
    qm_method: str,
    datasets: Mapping[str, object],
    *,
    skip_if_equal_to_base: bool = True,
) -> bool:
    if not raw_fea_path.exists():
        return False

    suffix = method_suffix(qm_method)
    with h5py.File(raw_fea_path, "a") as h5:
        if molname not in h5:
            return False
        grp = h5[molname]
        for base_name, data in datasets.items():
            suffixed_name = f"{base_name}{suffix}"
            arr = _as_array(data)
            if skip_if_equal_to_base and base_name in grp:
                base_arr = np.asarray(grp[base_name][()])
                if base_arr.shape == arr.shape and np.array_equal(base_arr, arr):
                    continue
            write_dataset(grp, suffixed_name, arr)
        grp.attrs[f"has_{qm_method}_datasets"] = True
    return True
