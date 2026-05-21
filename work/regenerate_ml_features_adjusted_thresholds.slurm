#!/bin/bash
#SBATCH --job-name=OSVMP2_Regenerate_1e-6
#SBATCH --partition=condo_amd
#SBATCH --qos=normal
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=/scr/u/u3651388/osv_mp2_ml_gen/work/logs/regenerate_ml_features_%a_%j.out
#SBATCH --error=/scr/u/u3651388/osv_mp2_ml_gen/work/logs/regenerate_ml_features_%a_%j.err
#SBATCH --array=0-0

ulimit -l unlimited

BASE_OFFSET=1
CHUNKSIZE=16000
CONCURRENCY=64

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
export OUTPUT_DIR="/lustre1/g/chem_yangjun/u3651388/osv_mp2_ml_gen/work/test/dsgdb9nsd_ccsdt_augccpvtz_adjusted_1e-6_${START_MOL}_${END_MOL}"

export HDF5_MERGE_MODE="overwrite"
export SLOT_BASE="/tmp/${USER}_run_slots/regen_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export STAGING_BASE="/tmp/${USER}_run_staging/regen_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${WORK_ROOT}/logs"

source "/scr/u/u3651388/osv_mp2_ml_gen/orca2pyscf/init_mokit.sh"

# PySCF and OSVMP2 configuration
export basis="aug-cc-pVTZ"
export auxbasis_hf="aug-cc-pvtz-jkfit"
export auxbasis_mp2="aug-cc-pvtz-ri"
export LOC_MISSING_POLICY="FAIL"

# OSVMP2 parameters
export local_type=1
export verbose=5
export solver=geometric
export shared_disk=0
export max_memory=10000
export method=osvmp2
export use_frozen=1
export use_sl=1
export cal_mode=mlmp2int
export save_hf_mat=0
export save_loc_mat=0
export wrap_test=0
export direct_int=0
export loc_fit=0
export spin=0
export cposv_tol=1e-10
export osv_tol=1e-4

# ========== CRITICAL: NEW THRESHOLDS TO MATCH ORCA ==========
export remo_tol=1e-6    # WAS: 1e-2 (old default)
export disc_tol=1e-6    # WAS: 1e-7 (old default)
# This aligns pair selection with ORCA's TCutPairs=1e-6
# =========================================================

export threeb_tol=0.2
export shell_tol=1e-10
export fit_tol=1e-6
export bfit_tol=1e-2
export save_pene=1
export basis_molpro=0
export charge_test=1
export ncore=1
export use_mbe=1
export use_ga=1

export TMPDIR="/tmp/${USER}_pyscf_tmp_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

echo "=========================================="
echo "OSVMP2 Regeneration with New Thresholds"
echo "=========================================="
echo "Molecules: ${START_MOL} - ${END_MOL}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""
echo "NEW PAIR SELECTION THRESHOLDS:"
echo "  remo_tol = ${remo_tol}  (OLD: 1e-2)"
echo "  disc_tol = ${disc_tol}  (OLD: 1e-7)"
echo ""
echo "This matches ORCA's TCutPairs=1e-6"
echo "=========================================="
echo ""

time python "${WORK_ROOT}/batch_osvmp2_runner.py" "${START_MOL}" "${END_MOL}" \
    --concurrency "${CONCURRENCY}" \
    --mode "${HDF5_MERGE_MODE}"

ML_FEA_H5="${OUTPUT_DIR}/ml_features.hdf5"
if [[ -f "${ML_FEA_H5}" ]]; then
    echo ""
    echo "✅ ml_features.hdf5 regenerated successfully:"
    echo "   Location: ${ML_FEA_H5}"
    
    # Quick validation: count groups
    N_GROUPS=$(h5dump -H "${ML_FEA_H5}" 2>/dev/null | grep "GROUP \"dsgdb9nsd" | wc -l)
    echo "   Number of molecule groups: ${N_GROUPS}"
else
    echo ""
    echo "❌ ml_features.hdf5 not created!"
fi

rm -rf "$TMPDIR" "$SLOT_BASE" "$STAGING_BASE"

echo ""
echo "=========================================="
echo "Job completed at $(date)"
echo "=========================================="
