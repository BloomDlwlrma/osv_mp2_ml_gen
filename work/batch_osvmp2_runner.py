#!/usr/bin/env python3
import os
import sys
import shutil
import argparse
import subprocess
import multiprocessing
import traceback
import re
import h5py
import uuid
import errno


def normalize_basis_for_suffix(basis):
    """Normalize basis string to checkpoint suffix style, e.g. 'aug-cc-pVTZ' -> 'augccpvtz'."""
    if not basis:
        return ""
    return re.sub(r"[^A-Za-z0-9]", "", basis).lower()


def infer_chk_suffix(chk_base, fallback_basis=None):
    """
    Infer checkpoint filename suffix from a base path name.
    Example: dsgdb9nsd_ccsdt_augccpvtz -> ccsdt_augccpvtz
    """
    base_name = os.path.basename(os.path.normpath(chk_base or ""))
    m = re.search(r"ccsdt_([A-Za-z0-9_+-]+)$", base_name)
    if m:
        return f"ccsdt_{normalize_basis_for_suffix(m.group(1))}"

    basis_norm = normalize_basis_for_suffix(fallback_basis)
    if basis_norm:
        return f"ccsdt_{basis_norm}"

    return "ccsdt_631gss"

# Setup workspace and path bases from environment (with safe defaults)
WORK_DIR = os.environ.get("WORK_DIR", os.path.dirname(os.path.abspath(__file__)))
ORCA2PYSCF_DIR = os.path.normpath(os.path.join(WORK_DIR, "..", "orca2pyscf"))
XYZ_BASE = os.environ.get("XYZ_BASE", os.path.join(ORCA2PYSCF_DIR, "xyz_files"))
CHK_HF_BASE = os.environ.get(
    "CHK_HF_BASE",
    os.path.join(ORCA2PYSCF_DIR, "source_files", "dsgdb9nsd_ccsdt_631gss"),
)
CHK_LOC_BASE = os.environ.get(
    "CHK_LOC_BASE",
    os.path.join(ORCA2PYSCF_DIR, "source_files", "dsgdb9nsd_ccsdt_631gss"),
)
SLOT_BASE = os.environ.get("SLOT_BASE")
if not SLOT_BASE:
    SLOT_BASE = os.path.join("/tmp", f"{os.environ.get('USER', 'tmp')}_run_slots")
STAGING_BASE = os.environ.get("STAGING_BASE", os.path.join(WORK_DIR, "run_staging"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(WORK_DIR, "test", "dsgdb9nsd"))
DEFAULT_MERGE_MODE = os.environ.get("HDF5_MERGE_MODE", "APPEND_MISSING").upper()
SUPPORTED_MERGE_MODES = {"OVERWRITE", "APPEND_MISSING"}
FAILED_LOG_DIR = os.path.join(OUTPUT_DIR, "failed_logs")
CHK_SUFFIX_HF = infer_chk_suffix(CHK_HF_BASE, os.environ.get("basis", ""))
CHK_SUFFIX_LOC = infer_chk_suffix(CHK_LOC_BASE, os.environ.get("basis", ""))
sys.path.insert(0, WORK_DIR)
try:
    from collect_hdf5 import collect_hdf5
except ImportError:
    raise ImportError("Cannot import collect_hdf5.py. Make sure it exists in WORK_DIR.")

# Size of chunk dirs like `1_16000`
CHUNK_SIZE = 16000
ENV_DEFAULTS = {
    "local_type": "1",
    "verbose": "5",
    "solver": "geometric",
    "shared_disk": "0",
    "max_memory": "10000",
    "method": "osvmp2",
    "use_frozen": "1",
    "use_sl": "1",
    "cal_mode": "mlmp2int",
    "save_hf_mat": "0",
    "save_loc_mat": "0",
    "wrap_test": "0",
    "direct_int": "0",
    "loc_fit": "0",
    "spin": "0",
    "cposv_tol": "1e-10",
    "osv_tol": "1e-4",
    "remo_tol": "1e-6",
    "disc_tol": "1e-6",
    "threeb_tol": "0.2",
    "shell_tol": "1e-10",
    "fit_tol": "1e-6",
    "bfit_tol": "1e-2",
    "save_pene": "1",
    "basis_molpro": "0",
    "charge_test": "1",
    "ncore": "1",
    "OMP_NUM_THREADS": "1",
    "use_mbe": "1",
    "use_ga": "1",
}


def parse_energies_from_run_log(log_path):
    """
    Parse RHF/MP2/Total energies from a per-molecule run log.
    Returns a tuple: (rhf_energy, mp2_corr_energy, total_energy), where each value can be None.
    """
    rhf_energy = None
    mp2_corr_energy = None
    total_energy = None

    rhf_pat = re.compile(r"^\s*RHF\s+energy\s*:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
    mp2_pat = re.compile(r"^\s*MP2\s+correlation\s+energy\s*:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
    total_pat = re.compile(r"^\s*Total\s+energy\s*:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")

    try:
        with open(log_path, "r") as f:
            for line in f:
                m_rhf = rhf_pat.match(line)
                if m_rhf:
                    rhf_energy = float(m_rhf.group(1))
                    continue

                m_mp2 = mp2_pat.match(line)
                if m_mp2:
                    mp2_corr_energy = float(m_mp2.group(1))
                    continue

                m_total = total_pat.match(line)
                if m_total:
                    total_energy = float(m_total.group(1))
    except Exception as e:
        print(f"[Parser] Warning: failed parsing energies from {log_path}: {e}")

    return rhf_energy, mp2_corr_energy, total_energy


def write_energy_record_hdf5(h5_path, mol_name, rhf_energy, mp2_corr_energy, total_energy, mode):
    """
    Write one molecule's energy record into HDF5 based on mode.
    Layout:
      /<mol_name>/RHF_energy
      /<mol_name>/MP2_correlation_energy
      /<mol_name>/Total_energy

    mode:
      - OVERWRITE: delete existing molecule group and rewrite.
        - APPEND_MISSING: keep existing molecule group and skip write.

    Returns one of: "written", "overwritten", "skipped".
    """
    with h5py.File(h5_path, "a") as fout:
        if mol_name in fout:
            if mode == "APPEND_MISSING":
                return "skipped"

            del fout[mol_name]
            status = "overwritten"
        else:
            status = "written"

        grp = fout.create_group(mol_name)
        grp.create_dataset("RHF_energy", data=rhf_energy)
        grp.create_dataset("MP2_correlation_energy", data=mp2_corr_energy)
        grp.create_dataset("Total_energy", data=total_energy)
        return status

def get_chunk_dir(mol_id):
    chunk_idx = (mol_id - 1) // CHUNK_SIZE
    start = chunk_idx * CHUNK_SIZE + 1
    end = start + CHUNK_SIZE - 1
    return f"{start}_{end}"


def fsync_file(path):
    if not os.path.exists(path):
        return
    with open(path, "rb") as fh:
        os.fsync(fh.fileno())


def validate_hdf5_quick(path):
    if not os.path.exists(path):
        return False
    if os.path.getsize(path) <= 2048:
        return False
    try:
        with h5py.File(path, "r") as fin:
            _ = list(fin.keys())
        return True
    except Exception:
        return False


def move_to_staging(src_path, dst_path):
    """
    Move file to staging path.
    Uses rename on same filesystem; falls back to copy+unlink across filesystems.
    """
    try:
        os.rename(src_path, dst_path)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        shutil.copy2(src_path, dst_path)
        os.unlink(src_path)


def archive_failed_log(log_path, mol_name, slot_id, returncode=None):
    """Copy a per-molecule run log into the shared failed_logs directory."""
    if not os.path.exists(log_path):
        return None

    os.makedirs(FAILED_LOG_DIR, exist_ok=True)
    suffix = f"_rc{returncode}" if returncode is not None else ""
    archive_name = f"{mol_name}_slot{slot_id}{suffix}.log"
    archive_path = os.path.join(FAILED_LOG_DIR, archive_name)
    shutil.copy2(log_path, archive_path)
    return archive_path


def load_allowed_ids(path):
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing passed-ids file: {path}")
    allowed = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                allowed.add(int(line))
            except ValueError:
                continue
    return allowed or set()

def worker(slot_id, task_queue, result_queue, base_env, allowed_ids=None):
    """
    Worker function to process one molecule at a time.
    Isolates run into dynamically generated slot_N folders.
    """
    slot_dir = os.path.join(SLOT_BASE, f"slot_{slot_id}")
    os.makedirs(slot_dir, exist_ok=True)
    os.makedirs(STAGING_BASE, exist_ok=True)
    
    while True:
        try:
            mol_id = task_queue.get(timeout=3)
        except multiprocessing.queues.Empty:
            break
            
        if mol_id == 'STOP':
            break

        if allowed_ids is not None and mol_id not in allowed_ids:
            continue
            
        mol_name = f"dsgdb9nsd_{mol_id:06d}"
        chunk_dir = get_chunk_dir(mol_id)
        
        xyz_file = os.path.join(XYZ_BASE, f"{mol_name}.xyz")
        chk_hf = os.path.join(CHK_HF_BASE, chunk_dir, f"hf_mat_{mol_name}_{CHK_SUFFIX_HF}.chk")
        chk_loc = os.path.join(CHK_LOC_BASE, chunk_dir, f"loc_var_{mol_name}_{CHK_SUFFIX_LOC}.chk")
        
        # Check if source files exist
        if not os.path.exists(xyz_file) or not os.path.exists(chk_hf) or not os.path.exists(chk_loc):
            print(f"[Worker {slot_id}] Warning: {mol_name} missing source files, skipping.")
            print(f"[Worker {slot_id}] Expected xyz={xyz_file}")
            print(f"[Worker {slot_id}] Expected chk_hf={chk_hf}")
            print(f"[Worker {slot_id}] Expected chk_loc={chk_loc}")
            continue 
            
        # Clear previous temporaries from the slot to prevent cross-contamination
        for f in os.listdir(slot_dir):
            path = os.path.join(slot_dir, f)
            try:
                if os.path.isfile(path): 
                    os.unlink(path)
                elif os.path.isdir(path): 
                    shutil.rmtree(path)
            except Exception as e:
                print(f"[Worker {slot_id}] Warning: error cleaning {path}: {e}")
                
        # Setup specific environment for opt_df.py based on run.sh definition
        env = os.environ.copy()
        env.update(base_env)
        env["chkfile_hf"] = chk_hf
        env["chkfile_loc"] = chk_loc
        env["molecule"] = mol_name
        env["moles"] = xyz_file
        
        cmd = ["python", os.path.join(WORK_DIR, "osvmp2", "opt_df.py"), xyz_file]
        run_log_path = os.path.join(slot_dir, f"run_{mol_name}.log")
        
        try:
            with open(run_log_path, "w") as log_f:
                subprocess.run(cmd, env=env, cwd=slot_dir, stdout=log_f, stderr=subprocess.STDOUT, check=True)

            rhf_energy, mp2_corr_energy, total_energy = parse_energies_from_run_log(run_log_path)
            if rhf_energy is None or mp2_corr_energy is None or total_energy is None:
                print(f"[Worker {slot_id}] Warning: {mol_name} energy parse incomplete from {run_log_path}")
            
            # Check the output targets.
            ml_feat = os.path.join(slot_dir, "ml_features.hdf5")
            pair_en = os.path.join(slot_dir, "pair_energy.hdf5")
            
            if os.path.exists(ml_feat) and os.path.exists(pair_en):
                # Move outputs safely to a staging area so the next loop in the worker doesn't delete them
                staging_dir = STAGING_BASE
                os.makedirs(staging_dir, exist_ok=True)

                uniq = uuid.uuid4().hex[:8]
                safe_ml_feat = os.path.join(staging_dir, f"ml_features_{mol_name}_{slot_id}_{uniq}.hdf5")
                safe_pair_en = os.path.join(staging_dir, f"pair_energy_{mol_name}_{slot_id}_{uniq}.hdf5")
                
                move_to_staging(ml_feat, safe_ml_feat)
                move_to_staging(pair_en, safe_pair_en)

                if not (validate_hdf5_quick(safe_ml_feat) and validate_hdf5_quick(safe_pair_en)):
                    print(f"[Worker {slot_id}] Warning: {mol_name} staged outputs failed integrity check")
                    continue
                    
                # Submit staged paths back to collector for safe sequential HDF5 concatenation
                result_queue.put({
                    'mol_id': mol_id,
                    'mol_name': mol_name,
                    'ml_feat': safe_ml_feat,
                    'pair_en': safe_pair_en,
                    'rhf_energy': rhf_energy,
                    'mp2_corr_energy': mp2_corr_energy,
                    'total_energy': total_energy,
                })
            else:
                  print(f"[Worker {slot_id}] Warning: {mol_name} compute finished but output HDF5 not found!")
                  archive_failed_log(run_log_path, mol_name, slot_id)
                 
        except subprocess.CalledProcessError as e:
              archived_path = archive_failed_log(run_log_path, mol_name, slot_id, e.returncode)
              if archived_path:
                  print(f"[Worker {slot_id}] Warning: {mol_name} failed with code {e.returncode}; log saved to {archived_path}")
              else:
                  print(f"[Worker {slot_id}] Warning: {mol_name} failed with code {e.returncode}")

def collector(result_queue, output_dir, merge_mode):
    """
    Sequential collector. Safe-locks and appends sub-results into single main HDF5.
    """
    os.makedirs(output_dir, exist_ok=True)
    master_feat = os.path.join(output_dir, "ml_features.hdf5")
    master_en = os.path.join(output_dir, "pair_energy.hdf5")
    master_energy = os.path.join(output_dir, "energy_record.hdf5")
    merge_lock = os.path.join(output_dir, ".merge_in_progress")
    merge_complete = os.path.join(output_dir, ".merge_complete")
    failed_merge_dir = os.path.join(output_dir, ".failed_merges")
    os.makedirs(failed_merge_dir, exist_ok=True)
    with open(merge_lock, "w") as f_lock:
        f_lock.write("MERGE_IN_PROGRESS\n")

    totals = {
        "processed": 0,
        "merged": 0,
        "failed": 0,
        "ml_written": 0,
        "ml_overwritten": 0,
        "ml_skipped": 0,
        "ml_conflicts": 0,
        "pair_written": 0,
        "pair_overwritten": 0,
        "pair_skipped": 0,
        "pair_conflicts": 0,
        "energy_written": 0,
        "energy_overwritten": 0,
        "energy_skipped": 0,
        "energy_conflicts": 0,
    }
    
    try:
        while True:
            res = result_queue.get()
            if res == 'STOP':
                break

            totals["processed"] += 1
            mol_name = res['mol_name']
            try:
                ml_stats = collect_hdf5([res['ml_feat']], master_feat, mode=merge_mode)
                pair_stats = collect_hdf5([res['pair_en']], master_en, mode=merge_mode)

                totals["ml_written"] += ml_stats["written"]
                totals["ml_overwritten"] += ml_stats["overwritten"]
                totals["ml_skipped"] += ml_stats["skipped"]
                totals["ml_conflicts"] += ml_stats["conflicts"]

                totals["pair_written"] += pair_stats["written"]
                totals["pair_overwritten"] += pair_stats["overwritten"]
                totals["pair_skipped"] += pair_stats["skipped"]
                totals["pair_conflicts"] += pair_stats["conflicts"]

                rhf_energy = res.get('rhf_energy')
                mp2_corr_energy = res.get('mp2_corr_energy')
                total_energy = res.get('total_energy')

                if rhf_energy is None or mp2_corr_energy is None or total_energy is None:
                    print(f"[Collector] Warning: energy record incomplete for {mol_name}, skip writing energy_record.hdf5")
                    energy_status = "skipped"
                else:
                    energy_status = write_energy_record_hdf5(
                        master_energy,
                        mol_name,
                        rhf_energy,
                        mp2_corr_energy,
                        total_energy,
                        merge_mode,
                    )

                if energy_status == "written":
                    totals["energy_written"] += 1
                elif energy_status == "overwritten":
                    totals["energy_overwritten"] += 1
                elif energy_status == "skipped":
                    totals["energy_skipped"] += 1

                partial_failed = bool(
                    ml_stats["conflicts"] or pair_stats["conflicts"]
                )
                if partial_failed:
                    print(
                        f"[Collector] Warning: partial merge for {mol_name}: "
                        f"ml(conflicts={ml_stats['conflicts']}), pair(conflicts={pair_stats['conflicts']}), "
                        f"energy(status={energy_status})"
                    )
                else:
                    totals["merged"] += 1

                fsync_file(master_feat)
                fsync_file(master_en)
                fsync_file(master_energy)

                # Remove staged files only after successful merge + fsync.
                for stage_file in [res['ml_feat'], res['pair_en']]:
                    try:
                        if os.path.exists(stage_file):
                            os.remove(stage_file)
                    except Exception as exc:
                        print(f"[Collector] Warning: failed to remove staged file {stage_file}: {exc}")

            except Exception as e:
                totals["failed"] += 1
                print(f"[Collector] Warning: failed to collect {mol_name}: {e}")
                traceback.print_exc()
                for stage_file in [res.get('ml_feat'), res.get('pair_en')]:
                    if stage_file and os.path.exists(stage_file):
                        try:
                            dst = os.path.join(failed_merge_dir, os.path.basename(stage_file))
                            shutil.move(stage_file, dst)
                        except Exception as move_exc:
                            print(f"[Collector] Warning: failed to quarantine staged file {stage_file}: {move_exc}")
    finally:
        try:
            if os.path.exists(merge_lock):
                os.remove(merge_lock)
            with open(merge_complete, "w") as f_done:
                f_done.write("MERGE_COMPLETE\n")
        except Exception as exc:
            print(f"[Collector] Warning: failed to write merge sentinel: {exc}")

    total_conflicts = totals["ml_conflicts"] + totals["pair_conflicts"] + totals["energy_conflicts"]
    if totals["failed"] or total_conflicts:
        print(
            "[Collector] Warning summary: "
            f"processed={totals['processed']}, merged={totals['merged']}, failed={totals['failed']}, "
            f"conflicts={total_conflicts}"
        )


def count_existing_molecules(h5_path):
    """Count top-level keys if a master HDF5 exists."""
    if not os.path.exists(h5_path):
        return 0
    try:
        with h5py.File(h5_path, "r") as f:
            return len(f.keys())
    except OSError as e:
        print(f"[Runner] Warning: cannot read existing HDF5 '{h5_path}': {e}")
        return -1

def main():
    parser = argparse.ArgumentParser(description="Batch process OSVMP2 feature generation")
    parser.add_argument("start_id", type=int, help="Starting molecule ID (e.g. 1)")
    parser.add_argument("end_id", type=int, help="Ending molecule ID")
    parser.add_argument("--concurrency", type=int, default=16, help="Number of parallel calculate slots")
    parser.add_argument(
        "--passed-ids-file",
        default=os.environ.get("PASSED_IDS_FILE", ""),
        help="Optional allowlist of valid molecule IDs to generate",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MERGE_MODE,
        help="HDF5 merge mode: OVERWRITE, APPEND_MISSING",
    )
    args = parser.parse_args()

    merge_mode = args.mode.upper()
    if merge_mode not in SUPPORTED_MERGE_MODES:
        modes = ", ".join(sorted(SUPPORTED_MERGE_MODES))
        raise ValueError(f"Unsupported --mode '{args.mode}'. Use one of: {modes}")

    allowed_ids = load_allowed_ids(args.passed_ids_file)
    
    base_env = {key: os.environ.get(key, default) for key, default in ENV_DEFAULTS.items()}
    base_env.update(
        {
            "OSVMP2PATH": WORK_DIR,
            "PYTHONPATH": f"{WORK_DIR}:{os.environ.get('PYTHONPATH', '')}",
            "basis": os.environ.get("basis", "6-31G**"),
            "auxbasis_hf": os.environ.get("auxbasis_hf", "def2-svp-jkfit"),
            "auxbasis_mp2": os.environ.get("auxbasis_mp2", "def2-svp-ri"),
            "LOC_MISSING_POLICY": os.environ.get("LOC_MISSING_POLICY", "FAIL"),
            "LOC_VALIDATE_TOL": os.environ.get("LOC_VALIDATE_TOL", "1e-6"),
        }
    )
    
    manager = multiprocessing.Manager()
    task_queue = manager.Queue()
    result_queue = manager.Queue()
    
    # 1. Fill Queue
    for mol_id in range(args.start_id, args.end_id + 1):
        task_queue.put(mol_id)
        
    # Put Stop marker for each worker
    for _ in range(args.concurrency):
        task_queue.put('STOP')
        
    # 2. Launch single Collector target
    output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    existing_feat = count_existing_molecules(os.path.join(output_dir, "ml_features.hdf5"))
    existing_pair = count_existing_molecules(os.path.join(output_dir, "pair_energy.hdf5"))
    existing_energy = count_existing_molecules(os.path.join(output_dir, "energy_record.hdf5"))

    col_proc = multiprocessing.Process(target=collector, args=(result_queue, output_dir, merge_mode))
    col_proc.start()
    
    # 3. Launch processing workers
    workers = []
    for i in range(args.concurrency):
        p = multiprocessing.Process(target=worker, args=(i, task_queue, result_queue, base_env, allowed_ids))
        p.start()
        workers.append(p)
        
    # Wait for all workers to finish
    for p in workers:
        p.join()
        
    # Stop collector safely after workers are dead
    result_queue.put('STOP')
    col_proc.join()
    
    if existing_feat == -1 or existing_pair == -1 or existing_energy == -1:
        print("[Runner] Warning: baseline count failed for at least one output HDF5 file")

if __name__ == "__main__":
    main()
