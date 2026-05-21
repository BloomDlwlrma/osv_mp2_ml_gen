#!/usr/bin/env python3
"""
count_subdirs.py
Count orca2pyscf chk outputs and report missing files.

Examples:
    python count_subdirs.py --help
    python count_subdirs.py --start_mol 16001 --end_mol 32000
    python count_subdirs.py --methods ccsdt --basis 631gss --save_missing missing_ids.txt
"""

import argparse
import os
import re
import sys

ID_WIDTH = 6

def parse_args():
    parser = argparse.ArgumentParser(
        description="Count orca2pyscf chk files and report missing molecules."
    )
    parser.add_argument(
        "--source_root",
        type=str,
        default="/XYFS02/HDD_POOL/hku2021_fos4/hku2021_fos4xy_2/sherwin/orca_output/orca2pyscf/source_files",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="ccsdt",
    )
    parser.add_argument(
        "--basis",
        type=str,
        default="631gss",
    )
    parser.add_argument(
        "--start_mol",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--end_mol",
        type=int,
        default=133885,
    )
    parser.add_argument(
        "--save_missing",
        type=str,
        default="/XYFS02/HDD_POOL/hku2021_fos4/hku2021_fos4xy_2/sherwin/orca_output/orca2pyscf/source_files/missing_ids.txt",
    )
    return parser.parse_args()


def chunk_dir_name(mol_id):
    chunk_idx = (mol_id - 1) // 16000
    start = chunk_idx * 16000 + 1
    end = (chunk_idx + 1) * 16000
    return f"{start}_{end}"


def build_expected_paths(base_dir, methods, basis, mol_id):
    chunk_name = chunk_dir_name(mol_id)
    mol_id_str = f"{mol_id:0{ID_WIDTH}d}"
    mol_name = f"dsgdb9nsd_{mol_id_str}_{methods}_{basis}.chk"
    hf_name = f"hf_mat_{mol_name}"
    loc_name = f"loc_var_{mol_name}"
    chunk_dir = os.path.join(base_dir, chunk_name)
    return os.path.join(chunk_dir, hf_name), os.path.join(chunk_dir, loc_name)


def collect_discovered_ids(base_dir, methods, basis):
    discovered_ids = set()
    hf_re = re.compile(rf"^hf_mat_dsgdb9nsd_(\d+)_{re.escape(methods)}_{re.escape(basis)}\.chk$")
    loc_re = re.compile(rf"^loc_var_dsgdb9nsd_(\d+)_{re.escape(methods)}_{re.escape(basis)}\.chk$")

    for root, _, files in os.walk(base_dir):
        for fname in files:
            match = hf_re.match(fname) or loc_re.match(fname)
            if match:
                discovered_ids.add(int(match.group(1)))

    return discovered_ids


def apply_range_filter(ids, start_mol=None, end_mol=None):
    filtered = []
    for mid in ids:
        if start_mol is not None and mid < start_mol:
            continue
        if end_mol is not None and mid > end_mol:
            continue
        filtered.append(mid)
    return sorted(filtered)


def main():
    args = parse_args()

    base_dir = os.path.join(
        os.path.abspath(args.source_root), f"dsgdb9nsd_{args.methods}_{args.basis}"
    )

    if args.start_mol is not None and args.end_mol is not None and args.start_mol > args.end_mol:
        print("Error: --start_mol cannot be larger than --end_mol", file=sys.stderr)
        return 1

    if not os.path.isdir(base_dir):
        print(f"Error: output directory does not exist: {base_dir}", file=sys.stderr)
        return 1

    discovered_ids = collect_discovered_ids(base_dir, args.methods, args.basis)

    # If full range is given, scan the explicit universe so missing-both can be detected.
    if args.start_mol is not None and args.end_mol is not None:
        target_ids = list(range(args.start_mol, args.end_mol + 1))
    else:
        target_ids = apply_range_filter(discovered_ids, args.start_mol, args.end_mol)

    complete = 0
    missing = 0
    missing_ids = []
    missing_details = []

    for mol_id in target_ids:
        hf_path, loc_path = build_expected_paths(base_dir, args.methods, args.basis, mol_id)
        has_hf = os.path.isfile(hf_path)
        has_loc = os.path.isfile(loc_path)

        if has_hf and has_loc:
            complete += 1
            continue

        missing += 1
        missing_ids.append(mol_id)

        if not has_hf:
            missing_details.append(hf_path)
        if not has_loc:
            missing_details.append(loc_path)

    print(f"Scan directory: {base_dir}")
    print(f"Scanned molecules: {len(target_ids)}")
    print(f"Complete molecules: {complete}")
    print(f"Missing molecules: {missing}")

    if missing_ids:
        print("Missing IDs:")
        for mid in missing_ids:
            print(f"{mid:0{ID_WIDTH}d}")
        print("Missing files:")
        for fpath in missing_details:
            print(fpath)
    else:
        print("Missing IDs: (none)")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_logs_dir = os.path.join(script_dir, "logs")
    os.makedirs(default_logs_dir, exist_ok=True)

    if args.save_missing is None:
        if args.start_mol is not None and args.end_mol is not None:
            save_missing_path = os.path.join(
                default_logs_dir,
                f"missing_ids_{args.methods}_{args.basis}_{args.start_mol:0{ID_WIDTH}d}_{args.end_mol:0{ID_WIDTH}d}.txt",
            )
        else:
            save_missing_path = os.path.join(
                default_logs_dir,
                f"missing_ids_{args.methods}_{args.basis}.txt",
            )
    else:
        save_missing_path = args.save_missing

    with open(save_missing_path, "w", encoding="utf-8") as fout:
        for mid in missing_ids:
            fout.write(f"{mid:0{ID_WIDTH}d}\n")
    print(f"Saved missing IDs to: {save_missing_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
