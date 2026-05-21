# OSV-MP2 / OSV-CCSD Working Directory Core Documentation

This directory implements a complete quantum chemistry workflow that starts from ORCA result retrieval, continues with PySCF/OSV calculations, and ends with HDF5 aggregation and training data production. Its core goal is not to reinvent electronic structure theory, but to reliably integrate ORCA, MOKIT, PySCF, and the OSV feature pipeline, so that MP2/CCSD correlation energies can be extracted, validated, and reused in batch.

---

## 1. Core Contents of `osvmp2`

### 1.1 What Problem Does It Solve?

`osvmp2` is the main line of this workflow. It addresses **local electron correlation** problems. The core idea is to decompose the conventional MP2 correlation energy into orbital pair contributions, and then further compress the virtual space using **Orbital-Specific Virtuals (OSV)** so that the correlation contribution of each orbital pair can be expressed in a small local subspace.

From the code structure, the `osvmp2/` directory contains several typical components:

- `opt_df.py`, `hf_ene.py`, `hf_grad.py`: handle HF/DF solutions, energies and gradients.
- `OSVL.py`, `CPOSV.py`, `ZCPL.py`, `osvutil.py`: focus on OSV construction, localization and utility functions.
- `grad_addons.py`, `ga_addons.py`, `mbe_addons.py`: provide auxiliary logic for gradients, geometry optimizations and many‑body expansions.
- `loc/`, `berny/`, `geometric/`: support geometry optimization and local orbital handling.

### 1.2 Physical & Methodological Main Line

The methodology of `osvmp2` can be summarized in three layers:

1. **HF as reference wavefunction**: first obtain a self‑consistent field solution, giving occupied orbitals and the Fock structure.
2. **OSV as compressed representation**: for each occupied orbital, build an orbital‑specific virtual space, keeping only the most important virtual directions, thereby reducing the dimensionality of the MP2 correlation calculation.
3. **Pair‑wise accumulation of correlation energy**: the correlation energy is not computed globally at once, but organized on a per‑pair basis, which facilitates partitioning, parallelization, and subsequent machine learning modeling.

In other words, the key point of `osvmp2` is not “compute a single total MP2 energy”, but rather **structure the correlation energy into storable, partitionable, and learnable pair‑level data**.

### 1.3 Workflow Corresponding to the Code Implementation

From `work/batch_osvmp2_runner.py`, we see that the actual execution is batch‑oriented:

- For each molecule, locate its `xyz`, `hf_mat_*.chk`, `loc_var_*.chk` files.
- Run `osvmp2/opt_df.py` to generate `ml_features.hdf5` and `pair_energy.hdf5`.
- Parse from the calculation log: `RHF energy`, `MP2 correlation energy`, `Total energy`.
- Results are first written to a per‑worker temporary slot, then moved to a separate staging directory, and finally merged sequentially into the main HDF5 by a collector.

The design emphasis here is to **avoid race conditions caused by concurrent file deletion**. That is, workers compute quickly, and the collector merges safely.

**Representative code snippet (from `work/batch_osvmp2_runner.py`):**
```python
def move_to_staging(src_path, dst_path):
    try:
        os.rename(src_path, dst_path)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        shutil.copy2(src_path, dst_path)
        os.unlink(src_path)

def write_energy_record_hdf5(h5_path, mol_name, rhf_energy, mp2_corr_energy, total_energy, mode):
    with h5py.File(h5_path, 'a') as fout:
        if mol_name in fout:
            if mode == 'APPEND_MISSING':
                return 'skipped'
            del fout[mol_name]
        grp = fout.create_group(mol_name)
        grp.create_dataset('RHF_energy', data=rhf_energy)
        grp.create_dataset('MP2_correlation_energy', data=mp2_corr_energy)
        grp.create_dataset('Total_energy', data=total_energy)
        return 'written'

def worker(slot_id, task_queue, result_queue, base_env, allowed_ids=None):
    slot_dir = os.path.join(SLOT_BASE, f'slot_{slot_id}')
    os.makedirs(slot_dir, exist_ok=True)
    os.makedirs(STAGING_BASE, exist_ok=True)
    while True:
        try:
            mol_id = task_queue.get(timeout=3)
        except multiprocessing.queues.Empty:
            break
        if mol_id == 'STOP':
            break
        # prepare env and run opt_df.py in a per-slot temporary directory
        cmd = ['python', os.path.join(WORK_DIR, 'osvmp2', 'opt_df.py'), xyz_file]
        with open(run_log_path, 'w') as log_f:
            subprocess.run(cmd, env=env, cwd=slot_dir, stdout=log_f, stderr=subprocess.STDOUT, check=True)
        # parse energies and move outputs to staging for collector to merge
        move_to_staging(tmp_out, staging_out)
```
---

### 1.4 What is the output

The `osvmp2` pipeline ultimately produces three key types of data:

- `ml_features.hdf5`: local features for machine learning.
- `pair_energy.hdf5`: orbital-pair correlation energies.
- `energy_record.hdf5`: records of RHF / MP2 / total energy for each molecule.

These files form the foundation for subsequent training, validation, and error analysis.

### 1.5 Significance from a literature perspective

When understood in the context of OSV-MP2 literature, the core value of `osvmp2` lies in:

- Reducing the computational and storage cost of MP2 using local virtual spaces.
- Expressing correlation energies as a function of the molecule’s local structure, which is more suitable for extrapolation to large systems.
- Providing physically consistent labels for pair-based machine learning models.

This is also why this directory cares not only about “whether the calculation runs” but also about “whether the outputs can be reliably merged, validated, and reused.”

---

## 2. Brief description of `osvccsd`

`osvccsd` is a higher‑accuracy route that runs in parallel with `osvmp2`. Its goal is to extract **CCSD** and optionally **CCSD(T)** triples correction information from ORCA outputs, serving as higher‑level reference energies.

From `work/osvccsd/batch_osvccsd_runner.py` we can see its clear responsibilities:

- Batch‑read ORCA `.out` files.
- Call `01_extract_pair_corr_eng.py` to extract pair correlation energies.
- Generate two types of outputs depending on the mode (`osvccsd` or `osvccsdt`):
  - `pair_energy_osvccsd.hdf5`
  - `pair_energy_osvccsdt.hdf5`
- If the input is inconsistent with an existing master file, it first checks for missing molecules before deciding whether to proceed.

Thus, `osvccsd` can be understood as **a high‑accuracy reference supplement layer for OSV‑MP2**. It is not the main training target, but it is important for building more trustworthy benchmarks, analysing MP2 errors, and extending to higher‑accuracy goals.

---

## 3. Quick understanding of the workflow

```text
ORCA output / xyz
    ↓
MOKIT conversion + PySCF recovery
    ↓
osvmp2 computes local correlation features and pair energies
    ↓
HDF5 merging and validation
    ↓
Machine learning training / error analysis / high‑accuracy reference
