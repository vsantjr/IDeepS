# Project IDeepS

<img src="https://github.com/vsantjr/CAP/blob/master/Images/logo1ideeps.png" width=58%>

## Single Node/Single GPU

This configuration is the most basic in which you use only 1 node and only 1 GPU of this node. The submission script presented below is a general one, which can be used for all configurations. This script (download the [dcgan.srm](../Code/dcgan.srm)) was developed based on scripts and suggestions provided by [Pedro Santos](https://bit.ly/38e2Z4B) and [Eduardo Miranda](https://bit.ly/35ApPmg).

```
#!/bin/bash
#SBATCH --job-name sn-sg                # SLURM_JOB_NAME
#SBATCH --partition nvidia_small        # SLURM_JOB_PARTITION
#SBATCH --nodes=1                       # SLURM_JOB_NUM_NODES
#SBATCH --ntasks-per-node=1             # SLURM_NTASKS_PER_NODE
#SBATCH --cpus-per-task=10              # SLURM_CPUS_PER_TASK
#SBATCH --exclusive                     # Exclusive acccess to nodes
#SBATCH --no-requeue                    # No automated job resubmission
#SBATCH --time=01:00:00                 # Define execution time

# VARIABLES OF INTEREST IN THE SLURM ENVIRONMENT
# <https://slurm.schedmd.com/sbatch.html>
# SLURM_PROCID
#     The MPI rank (or relative process ID) of the current process.
# SLURM_LOCALID
#     Node local task ID for the process within a job.
# SLURM_NODEID
#     ID of the nodes allocated. 

echo '========================================'
echo '- Job ID:' $SLURM_JOB_ID
echo '- # of nodes in the job:' $SLURM_JOB_NUM_NODES
echo '- # of tasks per node:' $SLURM_NTASKS_PER_NODE
echo '- # of tasks:' $SLURM_NTASKS
echo '- # of cpus per task:' $SLURM_CPUS_PER_TASK
echo '- Dir from which sbatch was invoked:' ${SLURM_SUBMIT_DIR##*/}
echo -n '- Nodes allocated to the job: '
nodeset -e $SLURM_JOB_NODELIST

# Go to the working directory from which sbatch was invoked
cd $SLURM_SUBMIT_DIR

# Load gcc compiler
module load gcc/8.3

# Activate your conda environment (myenv below) where PyTorch is installed
source /scratch/proj/name.user/miniconda3/bin/activate
conda activate myenv

# Run your code
echo -n '<1. Starting python script > ' && date
echo '-- output -----------------------------'

srun python dcgan.py

echo '-- end --------------------------------'
echo -n '<2. quit>                    ' && date

```

In the submission script, Slurm parameter values must be preceded by #SBATCH. Some explanations about this script follow. 

The name of the job is ```sn-sg```, and the selected partition/queue is ```nvidia_small```. One node (```--nodes```) was selected and since the job will run on a B715 node, we have 2 available NVIDIA K40 GPUs. But in the parameter ```--ntasks-per-node``` we defined 1, and hence we will use only 1 GPU of the node. Slurm's default behavior is to restart/resubmit a job when one of the nodes fails due to, for example, runtime errors. To disable this functionality, we include the parameter ```no-requeue```. This is important because you will waste time (allocation units) from your project if you do not do that, due to the fact that Slurm will resubmit this job until the time that was estimated in ```--time``` was expired (in the case above, we selected the maximum allowed time for this queue, i.e. 1 hour).

It is necessary to load the gcc compiler (already installed in the SDumont supercomputer), and activate the conda environment where PyTorch is installed (```myenv``` in the case above). After that run your python code (```dcgan.py```).

In order to submit the ```dcgan.srm``` script do the following:

- Deactivate the ```myenv``` conda environment if it is already activated. The environment is activated within the script;
- run ```sbatch dcgan.srm```.

Click [here](../Code/dcgan.py) to access the ```dcgan.py``` program. Such a code was developed based on an official [PyTorch tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) created by [Nathan Inkawhich](https://github.com/inkawhich). It is a program addressing the deep convolutional generative adversarial network [(DCGAN)](https://arxiv.org/abs/1511.06434). Note that since the number of images in the target set is very small for DCGAN as well as the selected number of epochs, the outputs/results are not very promissing. See details in the code. 

If you want to see the notebook version of this code with detailed explanations, take a look [here](https://github.com/vsantjr/DeepLearningMadeEasy/blob/temp_23-09/PyTorch_DCGAN.ipynb).






## Author

[**Valdivino Alexandre de Santiago J&uacute;nior**](https://www.linkedin.com/in/valdivino-alexandre-de-santiago-j%C3%BAnior-103109206/?locale=en_US)

## Licence

This project is licensed under the GNU GENERAL PUBLIC LICENSE, Version 3 (GPLv3) - see the [LICENSE.md](../LICENSE) file for details.

## Cite

Please cite this repository if you use it as:

V. A. Santiago J&uacute;nior. Project IDeepS, 2022. Acessed on: *date of access*. Available: https://github.com/vsantjr/IDeepS. 