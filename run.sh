export local_type=1
export verbose=5
export solver=geometric
#export solver=berny
export shared_disk=0
#export max_memory=20
#export max_memory=36000

export method=osvmp2
export use_frozen=1
export use_sl=1
export wrap_test=0
export direct_int=0
export loc_fit=1
export outcore=1

export spin=0
export cposv_tol=1e-10
export osv_tol=1e-4
export remo_tol=1e-2
export disc_tol=1e-7
export threeb_tol=0.2
export shell_tol=1e-10
export fit_tol=1e-6
export bfit_tol=1e-2
export save_pene=1
export basis_molpro=0
export charge_test=0
export pop_method=meta_lowdin
export charge_method_mp2=cm5
export OMP_NUM_THREADS=1
export use_mbe=1
export use_ga=1
#########################################
#PATH OF PROGRAM
export OSVMP2PATH=/home/$USER/OSVMP2_test/work
source $OSVMP2PATH/test_functions.sh
export PYTHONPATH=$OSVMP2PATH:$PYTHONPATH

#########################################
#QM/MM
#export qm_atoms="[0,1,2]"
#export qm_region=$qm_atoms
#PARAMETERS
export cal_mode=ml_mp2int #energy, ml_mp2int
export basis=ccpvtz
export molecule=water2
export moles=/home/qjliang/work/coords/$molecule.xyz
export ncore=10
#########################################
#EXECUTE PROGRAM
mkdir -p test/temp;cd test/temp;rm -rf *tmp*;mpirun -n $ncore python $OSVMP2PATH/osvmp2/opt_df.py $moles
#sbatch sbatch-mpi.cmd
