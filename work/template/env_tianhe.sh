#!/bin/bash
#module load libcint/3.0.19-gcc-4.8.5
#module load python/2.7.14-gcc-4.8.5
source /WORK/app/toolshs/cnmodule.sh
module load Python/2.7.9-fPIC
export H5PY_DEFAULT_READONLY=1
export GLEX_USE_ZC_RNDV=0
export PYTHONPATH=/WORK/hku_che_juny_l/OSV-BOMD-TEST:$PYTHONPATH
#export LD_LIBRARY_PATH=/WORK/hku_che_juny_l/OSV-BOMD-TEST/pyscf/lib:$LD_LIBRARY_PATH
#export LIBRARY_PATH=/WORK/hku_che_juny_l/OSV-BOMD-TEST/pyscf/lib:$LIBRARY_PATH
#export MKLROOT=/WORK/app/intelcompiler/14.0.2/composer_xe_2013_sp1.2.144/mkl
export MKLROOT=/WORK/app/intel/composer_xe_2013_sp1.2.144/mkl
export LD_PRELOAD=$MKLROOT/lib/intel64/libmkl_def.so:$MKLROOT/lib/intel64/libmkl_core.so:$MKLROOT/lib/intel64/libmkl_avx2.so:$MKLROOT/lib/intel64/libmkl_intel_lp64.so:$MKLROOT/lib/intel64/libmkl_intel_thread.so:/WORK/app/intel/composer_xe_2013_sp1.2.144/compiler/lib/intel64/libiomp5.so
#module load anaconda2/4.2.0-gcc-4.8.5
#source activate mympi
#python -c "import mpi4py"

