export PYTHONPATH=/home/qjliang/OSV-BOMD-MPI/md:/home/qjliang/OSV-BOMD-MPI:$PYTHONPATH
export local_type=1
export verbose=5
export solver=geometric
#export solver=berny
export shared_disk=0
#export max_memory=20000
#export max_memory=36000

export cal_mode=grad
export use_frozen=1
export use_sl=1
export direct_int=1
export loc_fit=1

#export basis=def2-svp
export basis=def2-tzvp
#export basis=def2-qzvpp
#export basis=ccpvtz
#export basis=augccpvtz
#export basis=6-31G
#export basis=3-21G
#export basis=6-31G**
#export basis=6-31+g**

export spin=0
export cposv_tol=1e-10
#export idsvd_tol=1e-4
export osv_tol=1e-4
#export osv_tol=1e-10
#export osv_tol=0
export remo_tol=1e-2
export disc_tol=1e-7
export threeb_tol=0.2
export shell_tol=1e-10
export fit_tol=1e-6
export bfit_tol=1e-2

export method="rhf"

#For DFT on Gaussian
export func="b3lyp"

#For forcefields on Gaussian
export forcefield="amber"

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
export stride='1'
export nbeads='1'
export port=5`date +"%M%S"`
export seed=32225
export total_steps=20000
export time_step=0.5
export temperature=350


#For NVT and NPT
export tau=100
export therm_mode='langevin'
#For NPT
export pressure=10
export baro_mode='isotropic'

#Option of output: prop(properties), pos(trajectory), force, chk(checkpoint)
export output_opt="prop pos vel chk"
#####################################################################################
#export molecule=eigen.xyz;export charge=1;export basis=6-31+g**
#export molecule=zundel.xyz;export charge=1;export basis=6-31+g**
#export molecule=porphycene.xyz;export basis=6-31G*;export temp_ensem=1
export molecule=water04.xyz; export basis=6-31G
#; export template=/home/qjliang/OSV-BOMD-MPI/md/temp_water04.gjf
export ncore=10
md_test () {
    python3 gen_input.py $molecule
    nohup i-pi input.xml &
    export use_mbe=1; export use_ga=1; nohup mpirun -n $ncore python3 md_initializer.py $molecule &
    #export use_mbe=0; export use_ga=0; python3 md_initializer.py $mol
}
bach_test () {
    mol=$(echo $molecule | cut -d. -f1)
    rm -r "$mol"_*
    if [[ $method == "dft" || $method == "ff" ]]; then
        export OMP_NUM_THREADS=$ncore
        export ncore=1
        dir_i="$mol"_"$method"_"$basis"
        mkdir $dir_i; cp gen_input.py md_initializer.py $molecule $dir_i
        cd $dir_i;md_test;cd ..
    else
        if [[ $method == "rhf" || $method == "hf" ]]; then
            dir_i="$mol"_rhf
        else
            dir_i="$mol"_"$scr_tol"_"$threeb_tol"
        fi
        export port=5`date +"%M%S"`
        #export scr_tol=$i; export threeb_tol=$j
        mkdir $dir_i; cp gen_input.py md_initializer.py $molecule $dir_i
        cd $dir_i;md_test;cd ..
    fi
}

#md_test
#por_test
bach_test
