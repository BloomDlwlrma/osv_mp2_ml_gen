#!/bin/bash
#####################################################################################
###                                                                                 #
### slurm-mpi.cmd :					                            #				 
### A SLURM submission script for running MPI program in HPC2021 system             #
###                                                                                 #
### Usage:                                                                          #
###    cd to directory containing MPI program executable, then:                     #
###    sbatch <location of this script>/slurm-mpi.cmd                               #    
###                                                                                 #
###    You may modify the script to select different MPI libraries                  #
###      module load impi                 ( Intel MPI Libraries)                    #
###      module load openmpi              ( OpenMPI Libraries)                      #
###                                                                                 #
### - Written by Lilian Chan, HKU ITS (2021-3-2)                                    #
###                                                                                 #
#####################################################################################

#SBATCH --job-name=job                            # Job name
#SBATCH --mail-type=END,FAIL                      # Mail events
##SBATCH --mail-user=qliang@hku.hk                # Update your email address   
##SBATCH --qos=normal                              # Specific QOS (debug/normal/long)
#SBATCH --time=20-00:00:00                         # Wall time limit (days-hrs:min:sec)
#SBATCH --nodes=1                                 # Total number of compute node(s)
#SBATCH --ntasks=16                                # Total number of MPI tasks (i.e. processes)
#SBATCH --ntasks-per-node=16                       # Number of MPI tasks on each node
#SBATCH --cpus-per-task=1                         # Number of CPU threads per process
#SBATCH --mem=80G                                # Memory required
#SBATCH --output=slurm-%j.out                        # Standard output file  

#####################################################################################
### The following stuff will be executed in the first allocated node.               #
### Please don't modify it                                                          #
#####################################################################################
echo "SLURM_NTASKS          : $SLURM_NTASKS"
echo "SLURM_JOB_NUM_NODES   : $SLURM_JOB_NUM_NODES"
echo "SLURM_CPUS_PER_TASK   : $SLURM_CPUS_PER_TASK"
echo "SLURM_CPUS_ON_NODE    : $SLURM_CPUS_ON_NODE"
###################################################################################
cd ${SLURM_SUBMIT_DIR}
OUTFILE=${SLURM_JOB_NAME}.${SLURM_JOBID}
echo ===========================================================
echo "Job Start Time is `date "+%Y/%m/%d -- %H:%M:%S"`"



### The -n option is not required since mpirun will automatically determine from Slurm settings
#time mpirun ./helloworld-c-impi >> ${OUTFILE}
#time mpirun ./helloworld-f90-impi >> ${OUTFILE}

#source activate qjliang
export OSVMP2PATH=/home/qjliang/work
export PYTHONPATH=$OSVMP2PATH:$PYTHONPATH
fold=$(echo $folder | rev | cut -d/ -f1 | rev)
export dir_cal=/home/qjliang/product/test
dir_i=$dir_cal/`date +"%Y%m%d%H%M%S"`_"$fold"_"$method"_"$basis"
echo $dir_i
mkdir -p $dir_i
cd $dir_i

for moles in $folder/*.xyz; do
    rm -rf *tmp*;mpirun -n $ncore python $OSVMP2PATH/osvmp2/opt_df.py $moles > test.log
done

echo "Job Finish Time is `date "+%Y/%m/%d -- %H:%M:%S"`"

exit 0
