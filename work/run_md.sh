export cal_mode=grad
#For DFT on Gaussian
export func="b3lyp"

#Set up MD parameters
#Units
export time_unit=femtosecond
export temp_unit=kelvin
export potential_unit=electronvolt
export press_unit=bar
#megapascal
export cell_units=angstrom
export force_unit=piconewton

####################################################################################
#Parameters
export verbose=0
#export charge=1
export dyn_mode='nvt'
export stride='10'
export nbeads='1'
rand_one=${RANDOM:0-1}
rand_two=${RANDOM:0-1}
rand_thr=${RANDOM:0-1}
rand_fou=${RANDOM:0-1}
export port=5"$rand_one""$rand_two""$rand_thr""$rand_fou"
export seed=32225
export total_steps=50000
export time_step=1
export temperature=300

#For NVT and NPT
export tau=100
export therm_mode='langevin'
#For NPT
export pressure=10
export baro_mode='isotropic'

#Option of output: prop(properties), pos(trajectory), force, chk(checkpoint)
export output_opt="prop pos vel chk"
#####################################################################################
#export moles=eigen.xyz;export charge=1;export basis=6-31+g**
#export moles=zundel.xyz;export charge=1;export basis=6-31+g**
#export moles=porphycene.xyz;export basis=6-31G*;export temp_ensem=1
#export moles=water04.xyz; export basis=6-31G
#export chkfile_md=/home/qjliang/work/test/20220306013906_cg2_ch3cho2_anti_water471_cmbeosvmp2_ccpvdz/sim_nvt.chk
#export ncore=10
MDPATH=$OSVMP2PATH/osvmp2/md
md_test () {
    #export port=5`date +"%M%S"`
	rand_one=${RANDOM:0-1}
    rand_two=${RANDOM:0-1}
	rand_thr=${RANDOM:0-1}
	rand_fou=${RANDOM:0-1}
	export port=5"$rand_one""$rand_two""$rand_thr""$rand_fou"
    python $MDPATH/gen_input.py $moles
    i-pi input.xml &
    sleep 2
    export use_mbe=1; export use_ga=1; mpirun -n $ncore python $MDPATH/md_initializer.py $moles 
}
bach_test () {
    mol=$(echo $moles | rev | cut -d/ -f1 | cut -d. -f2 | rev)
    if [[ $method == "dft" || $method == "ff" ]]; then
        export OMP_NUM_THREADS=$ncore
        export ncore=1
    fi
    dir_i=`date +"%Y%m%d%H%M%S"`_"$mol"_"$method"_"$basis"
    mkdir -p test; mkdir test/$dir_i
    cd test/$dir_i; md_test; cd ..
}

#md_test
#por_test
bach_test
