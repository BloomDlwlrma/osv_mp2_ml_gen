#!/bin/bash
# Personal ORCA submit script
# To use this script, invoke 'make_orca_job ${your_job_name}' in your terminal
# The variable ${your_job_name} is the filename if ORCA input file without ".inp" suffix

# To make this script self-dependent, the bash function 'make_orca_job' is shown below.
# One can add it to your .bashrc or .bash_profile to use it.

# make_orca_job () {
# 	if [[ ! -f orca.sh ]];then
# 		cp $HOME/orca.sh .
# 	fi
# 
#	sed -i '/^#/!s/.*job=.*/job='"$1"'/g' orca.sh
# }

OPENMPIPATH=/data/openmpi-4.1.6
export PATH=$OPENMPIPATH/bin:$PATH
export LD_LIBRARY_PATH=$OPENMPIPATH/lib:$LD_LIBRARY_PATH
export OMPI_MCA_btl=^openib

ORCAPATH=/data/orca_6_0_1
export PATH=$ORCAPATH:$PATH
export LD_LIBRARY_PATH=$ORCAPATH:$LD_LIBRARY_PATH
orca=$ORCAPATH/orca

job=$1

time_submit=$(date +"%H:%M:%S %d/%m/%y")
time_start=$(date +"%s")
$orca ${job}.inp > ${job}.out
retval=$?
time_end=$(date +"%s")

time_elapsed=$((time_end - time_start))



make_record () {
    # collecting time
	h=$((time_elapsed/3600))
	m=$((time_elapsed%3600/60))
	s=$((time_elapsed%60))

	# initialize file
	record_file=job_record.txt
	> $record_file

	# make sure there is a new line after the Subject line
	# otherwise its following content will not be transferred
	echo -e "Subject: ORCA Job $job_state\n" >> $record_file
	echo -e "Job state    : $job_state" >> $record_file
	echo -e "Job host     : $HOSTNAME" >> $record_file
	echo -e "Job location : $PWD" >> $record_file

	echo -e "Job times    : $h hours $m minutes $s seconds" >> $record_file
	echo -e "Job submiited on $time_submit" >> $record_file
 }


 if [[ $retval -eq 0 ]];then
		job_state=DONE
 else
		job_state=FAILED
 fi
 make_record

