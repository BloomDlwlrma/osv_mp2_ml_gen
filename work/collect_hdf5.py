import h5py
import argparse
import numpy as np
import os
import shutil

__doc__ = 'Collect sliced hdf5 data into one single file'

SUPPORTED_MERGE_MODES = {"OVERWRITE", "APPEND_MISSING"}


def myargs():
    parser = argparse.ArgumentParser(prog='chdf5', description=__doc__)
    parser.add_argument(
        'input',
        nargs='+',
        help='One or more input HDF5 files to be collected'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='out.hdf5',
        help='Output HDF5 file'
    )
    parser.add_argument(
        '-O', '--overwrite',
        action='store_true',
        default=False,
        help='Overwrite existing data in output file'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default=None,
        help='Merge mode: OVERWRITE, APPEND_MISSING'
    )
    return parser.parse_args()


def _normalize_mode(mode=None, overwrite=None):
    """Normalize merge mode while preserving backward compatibility with overwrite flag."""
    if mode is not None:
        normalized = mode.upper()
    else:
        normalized = "OVERWRITE" if overwrite else "APPEND_MISSING"

    if normalized not in SUPPORTED_MERGE_MODES:
        raise ValueError(
            f"Unsupported merge mode '{normalized}'. "
            f"Supported modes: {', '.join(sorted(SUPPORTED_MERGE_MODES))}"
        )
    return normalized


def _dataset_values_equal(src_ds, dst_ds, rtol=1e-8, atol=1e-10):
    """Compare dataset values with tolerance for numeric dtypes."""
    if src_ds.shape != dst_ds.shape or src_ds.dtype != dst_ds.dtype:
        return False

    src_data = src_ds[()]
    dst_data = dst_ds[()]

    src_arr = np.asarray(src_data)
    dst_arr = np.asarray(dst_data)

    if np.issubdtype(src_arr.dtype, np.number) and np.issubdtype(dst_arr.dtype, np.number):
        return np.allclose(src_arr, dst_arr, rtol=rtol, atol=atol, equal_nan=True)
    return np.array_equal(src_arr, dst_arr)


def _copy_or_merge(source_file, dest, name, mode):
    """
    Recursively copy or merge an object from source_file to dest.
    """
    obj = source_file[name]

    if name in dest:
        if mode == "OVERWRITE":
            del dest[name]
            source_file.copy(obj, dest, name=name)
            return "overwritten"

        # APPEND_MISSING skips existing objects.
        return "skipped"

    source_file.copy(obj, dest, name=name)
    return "written"


def _validate_hdf5_file(input_path):
    if not os.path.exists(input_path):
        return False, "file not found"
    if os.path.getsize(input_path) <= 2048:
        return False, "file too small (<= 2KB), likely truncated"
    try:
        with h5py.File(input_path, 'r') as fin:
            _ = list(fin.keys())
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def _quarantine_file(input_path, output_path, reason):
    out_dir = os.path.dirname(os.path.abspath(output_path))
    quarantine_dir = os.path.join(out_dir, ".quarantine")
    os.makedirs(quarantine_dir, exist_ok=True)
    dst = os.path.join(quarantine_dir, os.path.basename(input_path))
    if os.path.exists(dst):
        base, ext = os.path.splitext(dst)
        idx = 1
        while os.path.exists(f"{base}_{idx}{ext}"):
            idx += 1
        dst = f"{base}_{idx}{ext}"
    shutil.move(input_path, dst)
    print(f"[collect_hdf5] Quarantined invalid input '{input_path}' -> '{dst}': {reason}")


def collect_hdf5(input_paths, output_path, overwrite=False, mode=None):
    """
    Merge multiple HDF5 files into a single output file.

    Returns structured stats dict:
      written, overwritten, skipped, conflicts, total_inputs, total_objects.
    """
    merge_mode = _normalize_mode(mode=mode, overwrite=overwrite)
    stats = {
        "written": 0,
        "overwritten": 0,
        "skipped": 0,
        "conflicts": 0,
        "total_inputs": 0,
        "total_objects": 0,
    }

    with h5py.File(output_path, 'a') as fout:
        for input_path in input_paths:
            is_valid, reason = _validate_hdf5_file(input_path)
            if not is_valid:
                stats["conflicts"] += 1
                _quarantine_file(input_path, output_path, reason)
                continue
            try:
                with h5py.File(input_path, 'r') as finp:
                    stats["total_inputs"] += 1
                    for key in finp:
                        status = _copy_or_merge(finp, fout, key, merge_mode)
                        if status not in stats:
                            raise RuntimeError(f"Unexpected merge status '{status}'")
                        stats[status] += 1
                        stats["total_objects"] += 1
            except OSError as e:
                raise ValueError(f"Failed to open input file '{input_path}': {e}") from e

    return stats


def main():
    args = myargs()
    collect_hdf5(args.input, args.output, overwrite=args.overwrite, mode=args.mode)


if __name__ == "__main__":
    main()
    # usage: chdf5 [-h] [-o OuTPUT] [-0] input [input
    # Collect sliced hdf5 data into one single file
    # positional arguments:
    #     input   One or more input HDF5 files to be collected
    # options:
    #     -h, --helpshow this help message and exit
    #     -o OUTPUT,--output OUTPUT
    #             Output HDF5 file
    #     -0,--overwrite Overwrite existing data in output file

