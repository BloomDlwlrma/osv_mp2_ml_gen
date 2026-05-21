export local_type=1
export verbose=5
export solver=geometric
#export solver=berny
export shared_disk=0
#export max_memory=20000
#export max_memory=36000
export max_memory=1

export method=osvmp2 # osvmp2
export use_frozen=1
export use_sl=1
export cal_mode=mlmp2int # method=osvmp2 + cal_mode=ml turn on HF feature

# following 2 switch only work when "ml" is in cal_mode
export save_hf_mat=1
export save_loc_mat=1

export wrap_test=0
export direct_int=0
export loc_fit=1

#export basis=def2-svp
#export basis=def2-tzvp
#export basis=def2-qzvpp
export basis="6-31g*"
#export basis=augccpvtz
#export basis=6-31G
#export basis=3-21G
#export basis=6-31G**
#export basis=6-31+g**

export spin=0
export cposv_tol=1e-10
#export idsvd_tol=1e-4
export osv_tol=1e-4
#export nosv_id=80
#export osv_tol=1e-10
#export osv_tol=0
export remo_tol=1e-2
export disc_tol=1e-7
export threeb_tol=0.2
export shell_tol=1e-10
export fit_tol=1e-6
export bfit_tol=1e-2
export save_pene=1
export basis_molpro=0
export charge_test=1
#source env.sh
export OSVMP2PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" #The path to the directory in which osvmp2 is put
export PYTHONPATH=$OSVMP2PATH:$PYTHONPATH
source $OSVMP2PATH/test_functions.sh
#export molecule=water5_chain_1.5
#export moles=$OSVMP2PATH/coords/water_chain/$molecule.xyz
#export molecule=water
#export moles=$OSVMP2PATH/coords/$molecule.xyz
#export chkfile_save=/home/qiujiang/Dropbox/code/MPI/work/chkfile/$molecule/$basis
export ncore=10;export OMP_NUM_THREADS=1;export use_mbe=1;export use_ga=1
#mkdir -p test/$molecule;cd test/$molecule;rm -rf *tmp*;mpirun -n $ncore python $OSVMP2PATH/osvmp2/opt_df.py $moles


N=20
for ((ii=2;ii<=N;ii=ii+2));do
	echo $ii
	export molecule=Gly_${ii}
	export Gly_path=$HOME/glycine
	export moles=$Gly_path/coords/$molecule.xyz
mkdir -p test/temp;cd test/temp;rm -rf *tmp*;mpirun -n $ncore python $OSVMP2PATH/osvmp2/opt_df.py $moles
    cd $OLDPWD
done
#fold=water_plane
#export folder=$OSVMP2PATH/coords/cg_test/$fold
#mkdir -p test/$fold
#cd test/$fold;loop_mol
