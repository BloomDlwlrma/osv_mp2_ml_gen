loop_mol () {
 	for moles in $folder/*.xyz; do
	    rm -rf *tmp*;mpirun -n $ncore python3 $OSVMP2PATH/osvmp2/opt_df.py $moles;rm -rf *tmp*
 	done
}

ml_cal () {
    #for moles in ./$folder/*.xyz; do
	for step in $(seq $step_init $step_disp $step_end);do
	    #mol=$(echo $moles | rev | cut -d/ -f1 | cut -d. -f2 | rev)
		#if [ $mol \> $init_step ];then
		echo $step
        moles="./$folder/$(printf "%05d" $step).xyz"
	    rm -rf *tmp*;mpirun -n $ncore python3 opt_df.py $moles
		#fi
    done
}

opt_test () {
    export conv=normal;export wrap_test=0
    for moles in ./$folder/*.xyz; do
        mol=$(echo $moles | rev | cut -d/ -f1 | cut -d. -f2 | rev)
		#rm -rf *tmp*;mpirun -n $ncore python3 opt_df.py $moles > "$mol"_opt.log
        for i in 1e-2 1e-3;do
            for j in 0.1 0.2;do
                export scr_tol=$i; export threeb_tol=$j; loop_mol >opt_"$mol"_"$scr_tol"_"$threeb_tol".log
            done
        done
 	done
}

thres_test () {
    export folder=ori_baker
	export local_type=0;export thres=1e-10
	export scr_tol=0;export threeb_tol=0;loop_mol >baker_rimp2.log
	export local_type=1;export thres=1e-4
	for i in 1e-1 1e-2 1e-3;do
		for j in 0.1 0.2 0.3;do
			export scr_tol=$i; export threeb_tol=$j; loop_mol >thres_"$scr_tol"_"$threeb_tol".log
		done
	done
}

finite_dif () {
    folder=finite_test
    cd $folder; rm *.xyz *xyz.log; cp $moles .
    python3 gen_xyz.py *.xyz;cd ..
    for moles in ./$folder/*.xyz; do
		rm -rf *tmp*;mpirun -n $ncore python3 opt_df.py $moles > $moles.log
 	done
    cd $folder; python3 read.py
}

pes_test () {
    folder=pes;export grad_cal=0
    for moles in ./$folder/*.xyz; do
		rm -rf *tmp*;mpirun -n $ncore python3 opt_df.py $moles > $moles.log
 	done
    cd $folder; python3 read_ene.py
}

idsvd_test () {
    for i in -1 0;do
        export idsvd_tol=$i;rm -rf *tmp*;mpirun -n $ncore python3 opt_df.py $moles >idsvd_"$idsvd_tol".log
    done
    for i in {4..8..1};do
        export idsvd_tol=1e-$i;rm -rf *tmp*;mpirun -n $ncore python3 opt_df.py $moles >idsvd_"$idsvd_tol".log
    done
}

cposv_test () {
    export cposv_tol=0;rm -rf *tmp*;mpirun -n $ncore python3 opt_df.py $moles >cposv_"$cposv_tol".log
    for i in {4..12..2};do
        export cposv_tol=1e-$i;rm -rf *tmp*;mpirun -n $ncore python3 opt_df.py $moles >cposv_"$cposv_tol".log
    done
}

loc_test () {
    export loc_fit=0;rm -rf *tmp*;mpirun -n $ncore python3 opt_df.py $moles >loctest_nolocfit.log
    export loc_fit=1
    for i in 1e-6 1e-5 1e-4;do
        export fit_tol=$i; rm -rf *tmp*;mpirun -n $ncore python3 opt_df.py $moles >loctest_"$fit_tol"_"$fit_tol_atom".log
    done
}

scal_size () {
    loop_mol > scal_size.log
}

scal_core () {
    for ncore in {12..96..12};do
        rm -rf *tmp*;mpirun -n $ncore python3 opt_df.py $moles >scal_core_$ncore.log
    done
}

md_batch () {
    if [[ $molecule == *"cg2"* ]]; then
        export qm_atoms="[0,1,2,3,4,5,6,7]"
        export cg_residue=CG2
    elif [[ $molecule == *"cg3"* ]]; then
        export qm_atoms="[0,1,2,3,4,5,6,7,8,9,10,11]"
        export cg_residue=CG3
    fi
    export nonwater_region=$qm_atoms
	for ((i=1;i<=ntraj;i++));do
        nohup bash run_md.sh &
        sleep 4
    done

}

md_loop () {
    for step in $(seq $step_init 1 $step_end);do
        step=`printf %03d $step`
        export molecule=cg2_ch3cho2_syn_water1112_"$step"
        export export moles=$OSVMP2PATH/coords/cg2_syn_water1112_md/$molecule.xyz
        if [[ $molecule == *"cg2"* ]]; then
            export qm_atoms="[0,1,2,3,4,5,6,7]"
            export cg_residue=CG2
        elif [[ $molecule == *"cg3"* ]]; then
            export qm_atoms="[0,1,2,3,4,5,6,7,8,9,10,11]"
            export cg_residue=CG3
        fi
        #export qm_region=$qm_atoms
        export nonwater_region=$qm_atoms
		unset chkfile_md
		#if [ -f $OSVMP2PATH/test/cg2_syn_water1112_md/$molecule.xyz ];then
		#	echo "$molecule is finished!!"
		if [ -d $OSVMP2PATH/test/*$molecule* ];then
			echo "$molecule is being computed!!"
		else
        	dir_chk=$(echo $OSVMP2PATH/test/backup/*$molecule*)
			dir_chk=$(echo $dir_chk | awk '{print $NF}')
        	if [ -d $dir_chk ];then
            	export chkfile_md=$dir_chk/sim_nvt.chk
            	echo "Use chkfile $chkfile_md"
        	fi
			bash run_md.sh
			sleep 1
		fi
    done
}
