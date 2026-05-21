#!/bin/bash
#SBATCH --job-name=mokit16k
#SBATCH --partition=condo_amd
#SBATCH --qos=normal
#SBATCH --time=1-00:00:00 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --output=./logs/mokit_%a_%j.out
#SBATCH --error=./logs/mokit_%a_%j.err
#SBATCH --array=0-7
# Load modules and environments
module purge

# --- Export infrastructure paths (override defaults when needed) ---
export WORK_ROOT="/scr/u/u3651388/osv_mp2_ml_gen"
export OUTPUT_ROOT="/lustre1/g/chem_yangjun/u3651388/qm9_reaction_eng/qm9_orca_work/qm9_orca_work_mole/orca_output"
export XYZ_ROOT="/lustre1/g/chem_yangjun/u3651388/osv_mp2_ml_gen/orca2pyscf/xyz_files"

# Set Working Directory (use WORK_ROOT)
WORK_DIR="${WORK_ROOT}/orca2pyscf"
cd "$WORK_DIR"

# Initialize MOKIT (source the separated script)
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
source "${WORK_DIR}/init_mokit.sh"

# Ensure logs directory exists
mkdir -p logs

# --- Configuration parameters ---
export METHOD=ccsdt
export BASIS_TAG=augccpvtz
export AO_BASIS=aug-cc-pVTZ
# Local orbital reconstruction control for ORCA -> PySCF conversion.
export ORCA_RECON_TOL=1e-4
export ORCA_RECON_POLICY=SKIP

# Determine molecule range for this array task
CHUNKSIZE=16000
START_BASE=8001
END_LIMIT=133885
ARRAY_ID="${SLURM_ARRAY_TASK_ID:-0}"
START_MOL=$(( START_BASE + ARRAY_ID * CHUNKSIZE ))
END_MOL=$(( START_MOL + CHUNKSIZE - 1 ))
if [[ "$END_MOL" -gt "$END_LIMIT" ]]; then
    END_MOL="$END_LIMIT"
fi
if [[ "$START_MOL" -gt "$END_LIMIT" ]]; then
    echo "No molecules left for conversion task ${ARRAY_ID}."
    exit 0
fi
export START_MOL=$START_MOL
export END_MOL=$END_MOL

# Run the Python script (Python reads WORK_ROOT/OUTPUT_ROOT/MOKIT_ROOT from exports above)
python "$WORK_DIR/workflow_orca2pyscf.py"
