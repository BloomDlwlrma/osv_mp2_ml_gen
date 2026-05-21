#!/bin/bash
#SBATCH --job-name=OSVMP2_OSVCCSD_Batch
#SBATCH --partition=condo_amd
#SBATCH --qos=normal
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=384G
#SBATCH --output=/scr/u/u3651388/osv_mp2_ml_gen/work/logs/h5gen_osvccsd_%a_%j.out
#SBATCH --error=/scr/u/u3651388/osv_mp2_ml_gen/work/logs/h5gen_osvccsd_%a_%j.err
#SBATCH --array=0-0
ulimit -l unlimited

BASE_OFFSET=1
CHUNKSIZE=16000
CONCURRENCY=128

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

START_MOL=$(( BASE_OFFSET + SLURM_ARRAY_TASK_ID * CHUNKSIZE ))
END_MOL=$(( START_MOL + CHUNKSIZE - 1 ))

export WORK_ROOT="/scr/u/u3651388/osv_mp2_ml_gen/work"
cd "$WORK_ROOT"

export XYZ_BASE="/lustre1/g/chem_yangjun/u3651388/osv_mp2_ml_gen/orca2pyscf/xyz_files"
export CHK_HF_BASE="/lustre1/g/chem_yangjun/u3651388/osv_mp2_ml_gen/orca2pyscf/source_files/dsgdb9nsd_ccsdt_augccpvtz"
export CHK_LOC_BASE="/lustre1/g/chem_yangjun/u3651388/osv_mp2_ml_gen/orca2pyscf/source_files/dsgdb9nsd_ccsdt_augccpvtz"
export ORCA_OUT_BASE="/lustre1/g/chem_yangjun/u3651388/qm9_reaction_eng/qm9_orca_work/qm9_orca_work_mole/orca_output/orca_out_ccsdt_augccpvtz"
export OUTPUT_DIR="/lustre1/g/chem_yangjun/u3651388/osv_mp2_ml_gen/work/test/dsgdb9nsd_ccsdt_augccpvtz_${START_MOL}_${END_MOL}"

export HDF5_MERGE_MODE="overwrite"  # Options: "overwrite", "append", "skip"
export SLOT_BASE="/tmp/${USER}_run_slots/batch_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export STAGING_BASE="/tmp/${USER}_run_staging/batch_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${WORK_ROOT}/logs"

source "/scr/u/u3651388/osv_mp2_ml_gen/orca2pyscf/init_mokit.sh"

# Export required PySCF and OSVMP2 variables
export basis="aug-cc-pVTZ"
export auxbasis_hf="aug-cc-pvtz-jkfit"
export auxbasis_mp2="aug-cc-pvtz-ri"
export LOC_MISSING_POLICY="FAIL" 

# Other environment settings for OSVMP2
export local_type=1
export verbose=5
export solver=geometric
export shared_disk=0
# Per-molecule reserved memory (MB)
export max_memory=10000
export method=osvmp2
export use_frozen=1
export use_sl=1
export cal_mode=mlmp2int
export save_hf_mat=0
export save_loc_mat=0

export TMPDIR="/tmp/${USER}_pyscf_tmp_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

TIMEFORMAT="Chunk Array ${START_MOL}-${END_MOL} elapsed=%E"

PAIR_H5="${OUTPUT_DIR}/pair_energy.hdf5"
ML_FEA_H5="${OUTPUT_DIR}/ml_features.hdf5"

if [[ -f "${PAIR_H5}" && -f "${ML_FEA_H5}" ]]; then
	echo "[osvccsd] Reusing existing pair_energy.hdf5 and ml_features.hdf5 in ${OUTPUT_DIR}"
else
	echo "[osvccsd] Missing pair_energy.hdf5/ml_features.hdf5 in ${OUTPUT_DIR}; running osvmp2 stage first"
	time python /scr/u/u3651388/osv_mp2_ml_gen/work/batch_osvmp2_runner.py "${START_MOL}" "${END_MOL}" --concurrency "${CONCURRENCY}" --mode "${HDF5_MERGE_MODE}"
fi

PAIR_INPUT_PATTERN="${ORCA_OUT_BASE}/${CHUNK_DIR}/**/*.out"
PAIR_OUTPUT_PATH="${PAIR_H5}"
RAW_FEA_PATH="${ML_FEA_H5}"

time python /scr/u/u3651388/osv_mp2_ml_gen/work/osvccsd/batch_osvccsd_runner.py "${PAIR_INPUT_PATTERN}" --output "${PAIR_OUTPUT_PATH}" --raw-fea "${RAW_FEA_PATH}" --merge-mode OVERWRITE

rm -rf "$TMPDIR" "$SLOT_BASE" "$STAGING_BASE"