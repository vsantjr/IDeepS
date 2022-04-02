# Project IDeepS

<img src="https://github.com/vsantjr/CAP/blob/master/Images/logo1ideeps.png" width=58%>

## Single Node/Single GPU

This configuration is the most basic one in which you use only 1 node and only 1 GPU of this node. The submission script presented below is a general one, which can be used for all configurations. This script (download the [dcgan.srm](../Code/dcgan.srm) script) was developed based on scripts and suggestions provided by [Pedro Santos](https://bit.ly/38e2Z4B) and [Eduardo Miranda](https://bit.ly/35ApPmg).

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

It is necessary to load the gcc compiler (already installed in the SDumont supercomputer), and activate the conda environment where PyTorch is installed (```myenv``` above). After that run your python code (```dcgan.py```).

In order to submit the ```dcgan.srm``` script do the following:

- Deactivate the ```myenv``` conda environment if it is already activated. The environment is activated within the script;
- run ```sbatch dcgan.srm```.

Click [here](../Code/dcgan.py) to access the ```dcgan.py``` program. Such a code was developed by Valdivino Santiago Júnior based on an official [PyTorch tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) created by [Nathan Inkawhich](https://github.com/inkawhich). It is a program addressing the deep convolutional generative adversarial network [(DCGAN)](https://arxiv.org/abs/1511.06434). Note that since the number of images in the target set is very small for DCGAN (only 1,294 images of the [imagenettetvt320](https://www.kaggle.com/datasets/valdivinosantiago/imagenettetvt320) dataset), the outputs/results are not very promissing. More details in the code. Furthermore, if you want to see the notebook version of this code with detailed explanations, take a look at [here](https://github.com/vsantjr/DeepLearningMadeEasy/blob/temp_23-09/PyTorch_DCGAN.ipynb). 

The output after running the program is show below.

```
========================================
- Job ID: 10481452
- # of nodes in the job: 1
- # of tasks per node: 1
- # of tasks: 1
- # of cpus per task: 10
- Dir from which sbatch was invoked: dcgan
- Nodes allocated to the job: sdumont3069
<1. Starting python script > Fri Apr  1 13:01:06 -03 2022
-- output -----------------------------
Random Seed:  999
Configuration: Single Node/Single GPU!
Available GPUs:  2
Image:  0  - Input shape:  torch.Size([128, 3, 64, 64])
Image:  1  - Input shape:  torch.Size([128])
+----------------+------------+
|    Modules     | Parameters |
+----------------+------------+
| main.0.weight  |   819200   |
| main.1.weight  |    512     |
|  main.1.bias   |    512     |
| main.3.weight  |  2097152   |
| main.4.weight  |    256     |
|  main.4.bias   |    256     |
| main.6.weight  |   524288   |
| main.7.weight  |    128     |
|  main.7.bias   |    128     |
| main.9.weight  |   131072   |
| main.10.weight |     64     |
|  main.10.bias  |     64     |
| main.12.weight |    3072    |
+----------------+------------+
Total trainable params: 3576704
Checking trainable parameters: 3576704
Generator(
  (main): Sequential(
    (0): ConvTranspose2d(100, 512, kernel_size=(4, 4), stride=(1, 1), bias=False)
    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
    (6): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (7): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (8): ReLU(inplace=True)
    (9): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (10): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (11): ReLU(inplace=True)
    (12): ConvTranspose2d(64, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (13): Tanh()
  )
)
+----------------+------------+
|    Modules     | Parameters |
+----------------+------------+
| main.0.weight  |    3072    |
| main.2.weight  |   131072   |
| main.3.weight  |    128     |
|  main.3.bias   |    128     |
| main.5.weight  |   524288   |
| main.6.weight  |    256     |
|  main.6.bias   |    256     |
| main.8.weight  |  2097152   |
| main.9.weight  |    512     |
|  main.9.bias   |    512     |
| main.11.weight |    8192    |
+----------------+------------+
Total trainable params: 2765568
Checking trainable parameters: 2765568
Discriminator(
  (main): Sequential(
    (0): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (1): LeakyReLU(negative_slope=0.2, inplace=True)
    (2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (4): LeakyReLU(negative_slope=0.2, inplace=True)
    (5): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (6): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): LeakyReLU(negative_slope=0.2, inplace=True)
    (8): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (9): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): LeakyReLU(negative_slope=0.2, inplace=True)
    (11): Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), bias=False)
    (12): Sigmoid()
  )
)
Starting Training Loop...
[W NNPACK.cpp:80] Could not initialize NNPACK! Reason: Unsupported hardware.
[0/20][0/11]	Loss_D: 1.9201	Loss_G: 5.0795	D(x): 0.3905 -- 0.3905	D(G(z1 / z2)): 0.4996 -- 0.4996 / 0.0090 -- 0.0090
[1/20][0/11]	Loss_D: 0.6286	Loss_G: 10.7728	D(x): 0.8646 -- 0.8646	D(G(z1 / z2)): 0.3111 -- 0.3111 / 0.0000 -- 0.0000
[2/20][0/11]	Loss_D: 0.4348	Loss_G: 12.7065	D(x): 0.7353 -- 0.7353	D(G(z1 / z2)): 0.0001 -- 0.0001 / 0.0000 -- 0.0000
[3/20][0/11]	Loss_D: 0.1444	Loss_G: 21.4066	D(x): 0.8819 -- 0.8819	D(G(z1 / z2)): 0.0000 -- 0.0000 / 0.0000 -- 0.0000
[4/20][0/11]	Loss_D: 0.5369	Loss_G: 16.2693	D(x): 0.9702 -- 0.9702	D(G(z1 / z2)): 0.3503 -- 0.3503 / 0.0000 -- 0.0000
[5/20][0/11]	Loss_D: 0.1424	Loss_G: 11.4901	D(x): 0.8976 -- 0.8976	D(G(z1 / z2)): 0.0000 -- 0.0000 / 0.0000 -- 0.0000
[6/20][0/11]	Loss_D: 0.5912	Loss_G: 14.5343	D(x): 0.7577 -- 0.7577	D(G(z1 / z2)): 0.0000 -- 0.0000 / 0.0000 -- 0.0000
[7/20][0/11]	Loss_D: 0.8440	Loss_G: 18.0960	D(x): 0.6795 -- 0.6795	D(G(z1 / z2)): 0.0000 -- 0.0000 / 0.0000 -- 0.0000
[8/20][0/11]	Loss_D: 0.8844	Loss_G: 24.5297	D(x): 0.6435 -- 0.6435	D(G(z1 / z2)): 0.0000 -- 0.0000 / 0.0000 -- 0.0000
[9/20][0/11]	Loss_D: 1.9352	Loss_G: 20.3484	D(x): 0.9633 -- 0.9633	D(G(z1 / z2)): 0.8286 -- 0.8286 / 0.0000 -- 0.0000
[10/20][0/11]	Loss_D: 0.2024	Loss_G: 11.4599	D(x): 0.8737 -- 0.8737	D(G(z1 / z2)): 0.0016 -- 0.0016 / 0.0000 -- 0.0000
[11/20][0/11]	Loss_D: 0.7395	Loss_G: 16.7392	D(x): 0.9073 -- 0.9073	D(G(z1 / z2)): 0.4279 -- 0.4279 / 0.0000 -- 0.0000
[12/20][0/11]	Loss_D: 1.6167	Loss_G: 4.1732	D(x): 0.4645 -- 0.4645	D(G(z1 / z2)): 0.0119 -- 0.0119 / 0.0245 -- 0.0245
[13/20][0/11]	Loss_D: 0.6391	Loss_G: 4.4016	D(x): 0.9038 -- 0.9038	D(G(z1 / z2)): 0.3931 -- 0.3931 / 0.0172 -- 0.0172
[14/20][0/11]	Loss_D: 0.5890	Loss_G: 5.7948	D(x): 0.9357 -- 0.9357	D(G(z1 / z2)): 0.3683 -- 0.3683 / 0.0050 -- 0.0050
[15/20][0/11]	Loss_D: 1.9434	Loss_G: 3.5776	D(x): 0.3087 -- 0.3087	D(G(z1 / z2)): 0.0081 -- 0.0081 / 0.0520 -- 0.0520
[16/20][0/11]	Loss_D: 0.7609	Loss_G: 5.6084	D(x): 0.9224 -- 0.9224	D(G(z1 / z2)): 0.4295 -- 0.4295 / 0.0075 -- 0.0075
[17/20][0/11]	Loss_D: 1.1227	Loss_G: 2.3297	D(x): 0.4961 -- 0.4961	D(G(z1 / z2)): 0.0271 -- 0.0271 / 0.1650 -- 0.1650
[18/20][0/11]	Loss_D: 1.1582	Loss_G: 1.9434	D(x): 0.5119 -- 0.5119	D(G(z1 / z2)): 0.1517 -- 0.1517 / 0.2056 -- 0.2056
[19/20][0/11]	Loss_D: 0.5718	Loss_G: 4.4363	D(x): 0.8258 -- 0.8258	D(G(z1 / z2)): 0.2607 -- 0.2607 / 0.0211 -- 0.0211
-- end --------------------------------
<2. quit>                    Fri Apr  1 13:07:20 -03 2022


```






## Author

[**Valdivino Alexandre de Santiago J&uacute;nior**](https://www.linkedin.com/in/valdivino-alexandre-de-santiago-j%C3%BAnior-103109206/?locale=en_US)

## Licence

This project is licensed under the GNU GENERAL PUBLIC LICENSE, Version 3 (GPLv3) - see the [LICENSE.md](../LICENSE) file for details.

## Cite

Please cite this repository if you use it as:

V. A. Santiago J&uacute;nior. Project IDeepS, 2022. Acessed on: *date of access*. Available: https://github.com/vsantjr/IDeepS. 