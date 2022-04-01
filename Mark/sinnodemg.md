# Project IDeepS

<img src="https://github.com/vsantjr/CAP/blob/master/Images/logo1ideeps.png" width=58%>

## Single Node/Multiple GPUs

In this configuration, we use only 1 node but all GPUs of this node. The submission script is shown below and can be accessed [here](../Code/singlemulti.srm).

```
#!/bin/bash
#SBATCH --job-name sn-mg                # SLURM_JOB_NAME
#SBATCH --partition nvidia_dev        # SLURM_JOB_PARTITION
#SBATCH --nodes=1                       # SLURM_JOB_NUM_NODES
#SBATCH --ntasks-per-node=2             # SLURM_NTASKS_PER_NODE
#SBATCH --cpus-per-task=10              # SLURM_CPUS_PER_TASK
#SBATCH --exclusive                     # Exclusive acccess to nodes
#SBATCH --no-requeue                    # No automated job resubmission
#SBATCH --time=00:20:00                 # Define execution time

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

This submission script is very similar to the one for the [single node/single GPU configuration](../Code/dcgan.srm) wit very few changes. The name of the job is ```sn-sg```, and the selected partition/queue is ```nvidia_dev```. One node (```--nodes```) was selected and since the job will run on a B715 node, we have 2 available NVIDIA K40 GPUs. We will now use both GPUs (```--ntasks-per-node = 2```). The ```--time``` parameter was set to 20 minutes since this is the maximum value for this queue. 

Click [here](../Code/dcgan.py) to access the ```dcgan.py``` program. Such a code was developed based on an official [PyTorch tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) created by [Nathan Inkawhich](https://github.com/inkawhich). It is a program addressing the deep convolutional generative adversarial network [(DCGAN)](https://arxiv.org/abs/1511.06434). Note that since the number of images in the target set is very small for DCGAN, the outputs/results are not very promissing. See details in the code. 

The output after running the program is show below.

```


```

If you want to see the notebook version of this code with detailed explanations, take a look [here](https://github.com/vsantjr/DeepLearningMadeEasy/blob/temp_23-09/PyTorch_DCGAN.ipynb).




## Author

[**Valdivino Alexandre de Santiago J&uacute;nior**](https://www.linkedin.com/in/valdivino-alexandre-de-santiago-j%C3%BAnior-103109206/?locale=en_US)

## Licence

This project is licensed under the GNU GENERAL PUBLIC LICENSE, Version 3 (GPLv3) - see the [LICENSE.md](../LICENSE) file for details.

## Cite

Please cite this repository if you use it as:

V. A. Santiago J&uacute;nior. Project IDeepS, 2022. Acessed on: *date of access*. Available: https://github.com/vsantjr/IDeepS. 