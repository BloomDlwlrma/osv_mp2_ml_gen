#!/usr/bin/env python3
import argparse
import glob
import json
import os
from typing import Dict, List, Tuple

import h5py
import numpy as np
from pyscf import gto

CHUNK_SIZE = 16000


def get_chunk_dir(mol_id: int) -> str:
    chunk_idx = (mol_id - 1) // CHUNK_SIZE
    start = chunk_idx * CHUNK_SIZE + 1
    end = start + CHUNK_SIZE - 1
    return f"{start}_{end}"


def normalize_basis_key(basis: str) -> str:
    return basis.lower().replace("-", "")


def read_xyz_coord(xyz_file: str) -> str:
    with open(xyz_file, "r") as f:
        lines = f.readlines()
    natom = int(lines[0])
    return "".join(lines[2:2 + natom])


def find_xyz(xyz_root: str, mol_name: str) -> str:
    direct = os.path.join(xyz_root, f"{mol_name}.xyz")
    if os.path.exists(direct):
        return direct
    matches = glob.glob(os.path.join(xyz_root, "**", f"{mol_name}.xyz"), recursive=True)
    return matches[0] if matches else ""


def safe_read_dataset(fin: h5py.File, path: str) -> np.ndarray:
    if path not in fin:
        raise KeyError(f"Missing dataset: {path}")
    return np.asarray(fin[path][()])


def validate_single(
    xyz_file: str,
    hf_chk: str,
    loc_chk: str,
    basis: str,
    tol: float,
) -> Tuple[bool, Dict[str, float], str]:
    if not os.path.exists(xyz_file):
        return False, {}, f"Missing xyz: {xyz_file}"
    if not os.path.exists(hf_chk):
        return False, {}, f"Missing hf chk: {hf_chk}"
    if not os.path.exists(loc_chk):
        return False, {}, f"Missing loc chk: {loc_chk}"

    coord = read_xyz_coord(xyz_file)
    mol = gto.Mole()
    mol.atom = coord
    mol.basis = basis
    mol.charge = int(os.environ.get("charge", 0))
    mol.spin = int(os.environ.get("spin", 0))
    mol.build(verbose=0)

    with h5py.File(hf_chk, "r") as fhf:
        mo_coeff = safe_read_dataset(fhf, "scf/mo_coeff")
        mo_occ = safe_read_dataset(fhf, "scf/mo_occ")

    with h5py.File(loc_chk, "r") as floc:
        o = safe_read_dataset(floc, "o")
        uo = safe_read_dataset(floc, "uo")
        loc_fock = safe_read_dataset(floc, "loc_fock")
        eo = safe_read_dataset(floc, "eo")

    if mo_coeff.shape[0] != mol.nao:
        return False, {}, f"AO mismatch: mo_coeff nao={mo_coeff.shape[0]}, mol.nao={mol.nao}"

    mo_occ_mask = mo_occ > 0
    no = int(np.sum(mo_occ_mask))
    mo_coeff_occ = mo_coeff[:, mo_occ_mask]

    if o.shape != mo_coeff_occ.shape:
        return False, {}, f"Shape mismatch for o: expected {mo_coeff_occ.shape}, got {o.shape}"
    if uo.shape != (no, no):
        return False, {}, f"Shape mismatch for uo: expected {(no, no)}, got {uo.shape}"
    if loc_fock.shape != (no, no):
        return False, {}, f"Shape mismatch for loc_fock: expected {(no, no)}, got {loc_fock.shape}"
    if eo.shape != (no,):
        return False, {}, f"Shape mismatch for eo: expected {(no,)}, got {eo.shape}"

    if not (np.all(np.isfinite(mo_coeff_occ)) and np.all(np.isfinite(o)) and np.all(np.isfinite(uo))):
        return False, {}, "Found non-finite values in AO/LO arrays"

    ovlp = mol.intor_symmetric("int1e_ovlp")
    eye_no = np.eye(no)
    ao_gram = mo_coeff_occ.T @ ovlp @ mo_coeff_occ
    ao_orth_mat = ao_gram - eye_no
    ao_orth_err = float(np.max(np.abs(ao_orth_mat)))
    ao_orth_rms_err = float(np.sqrt(np.mean(ao_orth_mat ** 2)))

    recon_mat = mo_coeff_occ @ uo - o
    recon_err = float(np.max(np.abs(recon_mat)))
    recon_rms_err = float(np.sqrt(np.mean(recon_mat ** 2)))
    recon_rel_fro_err = float(np.linalg.norm(recon_mat, ord="fro") / max(np.linalg.norm(o, ord="fro"), 1e-16))

    uo_orth_mat = uo.T @ uo - eye_no
    uo_orth_err = float(np.max(np.abs(uo_orth_mat)))
    uo_orth_rms_err = float(np.sqrt(np.mean(uo_orth_mat ** 2)))

    lo_gram = o.T @ ovlp @ o
    lo_orth_mat = lo_gram - eye_no
    lo_orth_err = float(np.max(np.abs(lo_orth_mat)))
    lo_orth_rms_err = float(np.sqrt(np.mean(lo_orth_mat ** 2)))

    loc_diag_diff = np.diag(loc_fock) - eo
    loc_diag_err = float(np.max(np.abs(loc_diag_diff)))
    loc_diag_rms_err = float(np.sqrt(np.mean(loc_diag_diff ** 2)))

    loc_sym_mat = loc_fock - loc_fock.T
    loc_sym_err = float(np.max(np.abs(loc_sym_mat)))
    loc_sym_rms_err = float(np.sqrt(np.mean(loc_sym_mat ** 2)))

    loc_offdiag = loc_fock - np.diag(np.diag(loc_fock))
    loc_offdiag_fro = float(np.linalg.norm(loc_offdiag, ord="fro"))

    try:
        uo_cond = float(np.linalg.cond(uo))
    except np.linalg.LinAlgError:
        uo_cond = float("inf")

    # Compare occupied and localized AO-space subspace projectors in S metric.
    p_occ = mo_coeff_occ @ mo_coeff_occ.T @ ovlp
    p_loc = o @ o.T @ ovlp
    subspace_proj_err = float(np.linalg.norm(p_occ - p_loc, ord="fro"))

    max_err = max(ao_orth_err, recon_err, uo_orth_err, loc_diag_err, loc_sym_err, lo_orth_err)

    metrics = {
        "nao": int(mol.nao),
        "no": no,
        "ao_orth_err": ao_orth_err,
        "ao_orth_rms_err": ao_orth_rms_err,
        "recon_err": recon_err,
        "recon_rms_err": recon_rms_err,
        "recon_rel_fro_err": recon_rel_fro_err,
        "uo_orth_err": uo_orth_err,
        "uo_orth_rms_err": uo_orth_rms_err,
        "uo_cond": uo_cond,
        "lo_orth_err": lo_orth_err,
        "lo_orth_rms_err": lo_orth_rms_err,
        "loc_diag_err": loc_diag_err,
        "loc_diag_rms_err": loc_diag_rms_err,
        "loc_sym_err": loc_sym_err,
        "loc_sym_rms_err": loc_sym_rms_err,
        "loc_offdiag_fro": loc_offdiag_fro,
        "subspace_proj_err": subspace_proj_err,
        "max_err": max_err,
    }

    if max_err > tol:
        return False, metrics, (
            f"Tolerance exceeded: max_err={max_err:.3e}, tol={tol:.3e}"
        )

    # Metadata consistency (if available)
    with h5py.File(loc_chk, "r") as floc:
        if "meta" in floc:
            meta = floc["meta"].attrs
            basis_key = str(meta.get("basis_key", ""))
            if basis_key and basis_key != normalize_basis_key(basis):
                return False, metrics, (
                    f"basis_key mismatch in loc chk: expected {normalize_basis_key(basis)}, got {basis_key}"
                )

    return True, metrics, "ok"


def main():
    parser = argparse.ArgumentParser(
        description="Batch validator for AO/local-orbital consistency between hf_mat.chk and loc_var.chk"
    )
    parser.add_argument("--xyz_root", required=True, help="Root directory for XYZ files")
    parser.add_argument("--hf_root", required=True, help="Root directory for HF chk files")
    parser.add_argument("--loc_root", required=True, help="Root directory for local-orbital chk files")
    parser.add_argument("--basis", default="6-31G**", help="Basis used to rebuild PySCF molecule")
    parser.add_argument("--start_mol", type=int, required=True, help="Start molecule id (inclusive)")
    parser.add_argument("--end_mol", type=int, required=True, help="End molecule id (inclusive)")
    parser.add_argument("--chk_suffix", default="_ccsdt_631gss.chk", help="CHK suffix after molecule name")
    parser.add_argument("--tol", type=float, default=1e-6, help="Absolute tolerance for checks")
    parser.add_argument("--output_json", default="ao_lo_validation_report.json", help="JSON report path")
    parser.add_argument(
        "--output_failed_json",
        default="",
        help="Optional failed-only JSON report path generated from the same validation pass",
    )
    parser.add_argument(
        "--save_mode",
        choices=["all", "failed-only"],
        default="all",
        help="Which molecule entries are written to JSON results list",
    )
    parser.add_argument("--strict", action="store_true", help="Return non-zero exit code if any molecule fails")
    args = parser.parse_args()

    results: List[Dict[str, object]] = []

    for mol_id in range(args.start_mol, args.end_mol + 1):
        mol_name = f"dsgdb9nsd_{mol_id:06d}"
        chunk = get_chunk_dir(mol_id)

        xyz_file = find_xyz(args.xyz_root, mol_name)
        hf_chk = os.path.join(args.hf_root, chunk, f"hf_mat_{mol_name}{args.chk_suffix}")
        loc_chk = os.path.join(args.loc_root, chunk, f"loc_var_{mol_name}{args.chk_suffix}")

        ok, metrics, message = validate_single(xyz_file, hf_chk, loc_chk, args.basis, args.tol)
        row = {
            "mol_id": mol_id,
            "mol_name": mol_name,
            "status": "ok" if ok else "failed",
            "message": message,
        }
        row.update(metrics)
        results.append(row)
        print(f"[{mol_name}] {'OK' if ok else 'FAILED'} - {message}")

    failed = [r for r in results if r["status"] != "ok"]
    summary = {
        "total": len(results),
        "passed": len(results) - len(failed),
        "failed": len(failed),
        "pass_rate": (len(results) - len(failed)) / max(len(results), 1),
        "tol": args.tol,
        "basis": args.basis,
        "save_mode": args.save_mode,
    }

    if args.save_mode == "failed-only":
        results_to_save = failed
    else:
        results_to_save = results

    summary["results_saved_count"] = len(results_to_save)
    summary["results"] = results_to_save
    summary["failed_mol_ids"] = [r["mol_id"] for r in failed]

    with open(args.output_json, "w") as fjson:
        json.dump(summary, fjson, indent=2)

    if args.output_failed_json:
        failed_summary = dict(summary)
        failed_summary["save_mode"] = "failed-only"
        failed_summary["results_saved_count"] = len(failed)
        failed_summary["results"] = failed
        with open(args.output_failed_json, "w") as fjson:
            json.dump(failed_summary, fjson, indent=2)

    print("=" * 80)
    print(f"Validation finished: total={summary['total']}, passed={summary['passed']}, failed={summary['failed']}")
    print(f"JSON report: {args.output_json}")
    if args.output_failed_json:
        print(f"Failed-only JSON report: {args.output_failed_json}")

    if args.strict and failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
