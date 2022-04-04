# Project IDeepS

<img src="https://github.com/vsantjr/CAP/blob/master/Images/logo1ideeps.png" width=58%>

## Single Node/Multiple GPUs

In this configuration, we use only 1 node but all GPUs of this node. The submission script is shown below and can be accessed [here](../Code/cnn_gpus_1n.srm).

```
#!/bin/bash
#SBATCH --job-name cnn_gpus_1n          # SLURM_JOB_NAME
#SBATCH --partition nvidia_dev          # SLURM_JOB_PARTITION
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

# Go to the work directory from which sbatch was invoked
cd $SLURM_SUBMIT_DIR

# Load gcc compiler
module load gcc/8.3

# Activate your conda environment (myenv below) where PyTorch is installed
source /scratch/proj/name.user/miniconda3/bin/activate
conda activate myenv

# Run your code
echo -n '<1. starting python script > ' && date
echo '-- output -----------------------------'

srun python cnn_gpus.py --epochs 5 --batch-size 16

echo '-- end --------------------------------'
echo -n '<2. quit>                    ' && date


```

This submission script is very similar to the one for the [single node/single GPU configuration](../Code/dcgan.srm) wit very few changes. The name of the job is ```cnn_gpus_1n```, and the selected partition/queue is ```nvidia_dev```. One node (```--nodes```) was selected and since the job will run on a B715 node, we have 2 available NVIDIA K40 GPUs. We will now use both GPUs (```--ntasks-per-node=2```). The ```--time``` parameter was set to 20 minutes since this is the maximum value for this queue. 

Click [here](../Code/cnn_gpus.py) to access the ```cnn_gpus.py``` program. This program was developed by Valdivino Santiago Júnior and [Eduardo Miranda](https://bit.ly/35ApPmg) based on recommendations from [IDRIS](http://www.idris.fr/eng/jean-zay/gpu/jean-zay-gpu-torch-multi-eng.html). It is a program that shows how to distribute a convolutional neural network (CNN) model implemented in PyTorch. Via Slurm, it allows to use multiple GPUs in a single node or in multiple nodes.

One important program imported in the ```cnn_gpus.py``` code is ```sdenv.py``` (click [here](../Code/sdenv.py) to access it). This program was also developed based on recommendations from IDRIS (http://www.idris.fr/eng/jean-zay/gpu/jean-zay-gpu-torch-multi-eng.html). Its goal is to expose the Slurm parameter values according to the subscription script (```.srm```) in order to run jobs in multiple GPUs. Import this file into your Python code. 

Moreover, we used the [imgnet320_c5](https://www.kaggle.com/datasets/valdivinosantiago/imgnet320-c5) dataset which is a subset of the imagenettetvt320 one with only 5 classes, and only the training and test datasets. But here, we used only the training dataset within the training phase.

The output after running the program is show below. We can clearly see that 2 GPUs were used and they are identified as ```Rank 0``` and ```Rank 1```.

```
========================================
- Job ID: 10483866
- # of nodes in the job: 1
- # of tasks per node: 2
- # of tasks: 2
- # of cpus per task: 10
- Dir from which sbatch was invoked: mnode
- Nodes allocated to the job: sdumont3076
<1. starting python script > Mon Apr  4 15:48:00 -03 2022
-- output -----------------------------
- Process 1 corresponds to GPU 1 of node 0
Rank:  1

Datasets size:  {'train': 4703}
Dataloaders size:  1
Training classes:  ['cassetePlayer', 'chainShaw', 'church', 'englishSpringer', 'frenchHorn']
Training lengths:  5
Height x Width: 224 x 224
@@@@ STARTING TRAINING! @@@@
>>> Training on  1  nodes and  2  processes, master node is  sdumont3076
- Process 0 corresponds to GPU 0 of node 0
Rank:  0

Datasets size:  {'train': 4703}
Dataloaders size:  1
Training classes:  ['cassetePlayer', 'chainShaw', 'church', 'englishSpringer', 'frenchHorn']
Training lengths:  5
Height x Width: 224 x 224
@@@@ STARTING TRAINING! @@@@
[W NNPACK.cpp:80] Could not initialize NNPACK! Reason: Unsupported hardware.
[W NNPACK.cpp:80] Could not initialize NNPACK! Reason: Unsupported hardware.
Epoch [1/5], Step [20/294], Loss: 1.5517, Time data load: 0.300ms, Time training: 161.769ms
Epoch [1/5], Step [40/294], Loss: 1.4656, Time data load: 0.286ms, Time training: 161.522ms
Epoch [1/5], Step [60/294], Loss: 1.0791, Time data load: 0.402ms, Time training: 161.986ms
Epoch [1/5], Step [80/294], Loss: 1.4699, Time data load: 0.713ms, Time training: 161.019ms
Epoch [1/5], Step [100/294], Loss: 1.6096, Time data load: 0.284ms, Time training: 162.648ms
Epoch [1/5], Step [120/294], Loss: 1.4987, Time data load: 0.273ms, Time training: 161.322ms
Epoch [1/5], Step [140/294], Loss: 1.6502, Time data load: 0.275ms, Time training: 161.086ms
Epoch [1/5], Step [160/294], Loss: 1.6963, Time data load: 0.289ms, Time training: 161.383ms
Epoch [1/5], Step [180/294], Loss: 1.9279, Time data load: 0.265ms, Time training: 161.530ms
Epoch [1/5], Step [200/294], Loss: 2.1645, Time data load: 0.304ms, Time training: 163.171ms
Epoch [1/5], Step [220/294], Loss: 1.0251, Time data load: 0.316ms, Time training: 162.261ms
Epoch [1/5], Step [240/294], Loss: 1.0725, Time data load: 0.283ms, Time training: 162.210ms
Epoch [1/5], Step [260/294], Loss: 0.9453, Time data load: 0.270ms, Time training: 161.860ms
Epoch [1/5], Step [280/294], Loss: 1.5071, Time data load: 0.321ms, Time training: 160.830ms
------- Before checkpoint 1
------- After checkpoint 1
Epoch [2/5], Step [20/294], Loss: 1.4192, Time data load: 0.319ms, Time training: 160.633ms
Epoch [2/5], Step [40/294], Loss: 1.4526, Time data load: 0.285ms, Time training: 160.728ms
Epoch [2/5], Step [60/294], Loss: 1.0225, Time data load: 0.329ms, Time training: 160.805ms
Epoch [2/5], Step [80/294], Loss: 1.2614, Time data load: 0.320ms, Time training: 160.725ms
Epoch [2/5], Step [100/294], Loss: 1.2396, Time data load: 0.308ms, Time training: 160.726ms
Epoch [2/5], Step [120/294], Loss: 1.5365, Time data load: 0.312ms, Time training: 161.181ms
Epoch [2/5], Step [140/294], Loss: 1.8403, Time data load: 0.369ms, Time training: 160.535ms
Epoch [2/5], Step [160/294], Loss: 1.3214, Time data load: 0.295ms, Time training: 160.715ms
Epoch [2/5], Step [180/294], Loss: 1.7228, Time data load: 0.337ms, Time training: 160.665ms
Epoch [2/5], Step [200/294], Loss: 1.7631, Time data load: 0.329ms, Time training: 160.937ms
Epoch [2/5], Step [220/294], Loss: 1.3702, Time data load: 0.356ms, Time training: 160.654ms
Epoch [2/5], Step [240/294], Loss: 1.0720, Time data load: 0.317ms, Time training: 161.177ms
Epoch [2/5], Step [260/294], Loss: 0.9634, Time data load: 0.307ms, Time training: 160.572ms
Epoch [2/5], Step [280/294], Loss: 1.6141, Time data load: 0.362ms, Time training: 160.518ms
------- Before checkpoint 2
------- After checkpoint 2
Epoch [3/5], Step [20/294], Loss: 1.0506, Time data load: 0.301ms, Time training: 160.901ms
Epoch [3/5], Step [40/294], Loss: 1.1300, Time data load: 0.284ms, Time training: 160.838ms
Epoch [3/5], Step [60/294], Loss: 0.9296, Time data load: 0.267ms, Time training: 161.079ms
Epoch [3/5], Step [80/294], Loss: 1.0273, Time data load: 0.270ms, Time training: 160.949ms
Epoch [3/5], Step [100/294], Loss: 1.1169, Time data load: 0.289ms, Time training: 161.252ms
Epoch [3/5], Step [120/294], Loss: 1.0694, Time data load: 0.272ms, Time training: 160.777ms
Epoch [3/5], Step [140/294], Loss: 1.9406, Time data load: 0.275ms, Time training: 160.803ms
Epoch [3/5], Step [160/294], Loss: 1.2758, Time data load: 0.282ms, Time training: 160.870ms
Epoch [3/5], Step [180/294], Loss: 1.6153, Time data load: 0.273ms, Time training: 161.296ms
Epoch [3/5], Step [200/294], Loss: 1.6608, Time data load: 0.268ms, Time training: 161.227ms
Epoch [3/5], Step [220/294], Loss: 1.0393, Time data load: 0.294ms, Time training: 160.915ms
Epoch [3/5], Step [240/294], Loss: 1.0962, Time data load: 0.273ms, Time training: 161.133ms
Epoch [3/5], Step [260/294], Loss: 0.7571, Time data load: 0.326ms, Time training: 160.844ms
Epoch [3/5], Step [280/294], Loss: 1.5394, Time data load: 0.293ms, Time training: 160.943ms
------- Before checkpoint 3
------- After checkpoint 3
Epoch [4/5], Step [20/294], Loss: 1.7168, Time data load: 0.301ms, Time training: 7.841ms
Epoch [4/5], Step [40/294], Loss: 1.6673, Time data load: 0.295ms, Time training: 160.974ms
Epoch [4/5], Step [60/294], Loss: 0.6337, Time data load: 0.245ms, Time training: 161.041ms
Epoch [4/5], Step [80/294], Loss: 0.8648, Time data load: 0.273ms, Time training: 160.722ms
Epoch [4/5], Step [100/294], Loss: 1.1576, Time data load: 0.292ms, Time training: 160.923ms
Epoch [4/5], Step [120/294], Loss: 1.0894, Time data load: 0.321ms, Time training: 7.652ms
Epoch [4/5], Step [140/294], Loss: 1.7190, Time data load: 0.268ms, Time training: 160.813ms
Epoch [4/5], Step [160/294], Loss: 1.2663, Time data load: 0.262ms, Time training: 161.252ms
Epoch [4/5], Step [180/294], Loss: 1.6760, Time data load: 0.288ms, Time training: 161.007ms
Epoch [4/5], Step [200/294], Loss: 1.7698, Time data load: 0.323ms, Time training: 160.894ms
Epoch [4/5], Step [220/294], Loss: 0.8896, Time data load: 0.354ms, Time training: 7.740ms
Epoch [4/5], Step [240/294], Loss: 1.0807, Time data load: 0.331ms, Time training: 161.583ms
Epoch [4/5], Step [260/294], Loss: 0.8706, Time data load: 0.297ms, Time training: 161.177ms
Epoch [4/5], Step [280/294], Loss: 1.6955, Time data load: 0.358ms, Time training: 160.896ms
------- Before checkpoint 4
------- After checkpoint 4
@@@@ END TRAINING - Rank: 1! @@@@
Epoch [5/5], Step [20/294], Loss: 1.6236, Time data load: 1.786ms, Time training: 159.451ms
Epoch [5/5], Step [40/294], Loss: 1.0767, Time data load: 0.288ms, Time training: 161.092ms
Epoch [5/5], Step [60/294], Loss: 1.0516, Time data load: 0.273ms, Time training: 161.137ms
Epoch [5/5], Step [80/294], Loss: 0.9370, Time data load: 0.300ms, Time training: 161.245ms
Epoch [5/5], Step [100/294], Loss: 0.8711, Time data load: 0.268ms, Time training: 160.913ms
Epoch [5/5], Step [120/294], Loss: 1.3299, Time data load: 0.266ms, Time training: 161.236ms
Epoch [5/5], Step [140/294], Loss: 1.3505, Time data load: 0.286ms, Time training: 160.932ms
Epoch [5/5], Step [160/294], Loss: 1.6415, Time data load: 0.275ms, Time training: 160.996ms
Epoch [5/5], Step [180/294], Loss: 1.6095, Time data load: 0.270ms, Time training: 161.045ms
Epoch [5/5], Step [200/294], Loss: 1.5528, Time data load: 0.270ms, Time training: 161.224ms
Epoch [5/5], Step [220/294], Loss: 0.9882, Time data load: 0.258ms, Time training: 161.361ms
Epoch [5/5], Step [240/294], Loss: 1.0285, Time data load: 0.273ms, Time training: 160.932ms
Epoch [5/5], Step [260/294], Loss: 0.8087, Time data load: 0.295ms, Time training: 161.138ms
Epoch [5/5], Step [280/294], Loss: 1.6912, Time data load: 0.274ms, Time training: 161.171ms
@@@@ END TRAINING - Rank: 0! @@@@
>>> Training complete in: 0:03:59.434408
-- end --------------------------------
<2. quit>                    Mon Apr  4 15:52:22 -03 2022
```

## Multiple Nodes/Multiple GPUs

In the last configuration, we require more GPUs than one node has available. Hence, we use several nodes all GPUs of each node. The submission script is shown below and can be accessed [here](../Code/cnn_gpus_nn.srm).

```
#!/bin/bash
#SBATCH --job-name cnn_gpus_nn          # SLURM_JOB_NAME
#SBATCH --partition nvidia_dev          # SLURM_JOB_PARTITION
#SBATCH --nodes=4                       # SLURM_JOB_NUM_NODES
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

# Go to the work directory from which sbatch was invoked
cd $SLURM_SUBMIT_DIR

# Load gcc compiler
module load gcc/8.3

# Activate your conda environment (myenv below) where PyTorch is installed
source /scratch/proj/name.user/miniconda3/bin/activate
conda activate myenv

# Run your code
echo -n '<1. starting python script > ' && date
echo '-- output -----------------------------'

srun python cnn_gpus.py --epochs 5 --batch-size 16

echo '-- end --------------------------------'
echo -n '<2. quit>                    ' && date
```

There are only two differences between this script for the multiple nodes/multiple GPUs configuration and the previous one for single node/multiple GPUs. Firstly, it is the name of the job. More importantly, now the number of nodes is 4 (```--nodes```) and the job will run on B715 nodes. Sine we have 2 available NVIDIA K40 GPUs per this type of node and we set to use both GPUs (```--ntasks-per-node=2```), then we will run this job in 8 NVIDIA K40 GPUs. The python codes are the same as in the single node configuration (```cnn_gpus.py```, ```sdenv.py```) as well as is the dataset (imgnet320_c5).

The output after running the program is show below. We can clearly see that 8 GPUs were used and they are identified as ```Rank 0```, ```Rank 1```, ..., ```Rank 7```.

```
========================================
- Job ID: 10483880
- # of nodes in the job: 4
- # of tasks per node: 2
- # of tasks: 8
- # of cpus per task: 10
- Dir from which sbatch was invoked: mnode
- Nodes allocated to the job: sdumont3076 sdumont3077 sdumont3078 sdumont3079
<1. starting python script > Mon Apr  4 15:58:03 -03 2022
-- output -----------------------------
- Process 4 corresponds to GPU 0 of node 2
Rank:  4

Datasets size:  {'train': 4703}
Dataloaders size:  1
Training classes:  ['cassetePlayer', 'chainShaw', 'church', 'englishSpringer', 'frenchHorn']
Training lengths:  5
Height x Width: 224 x 224
@@@@ STARTING TRAINING! @@@@
- Process 7 corresponds to GPU 1 of node 3
Rank:  7

Datasets size:  {'train': 4703}
Dataloaders size:  1
Training classes:  ['cassetePlayer', 'chainShaw', 'church', 'englishSpringer', 'frenchHorn']
Training lengths:  5
Height x Width: 224 x 224
@@@@ STARTING TRAINING! @@@@
- Process 2 corresponds to GPU 0 of node 1
Rank:  2

Datasets size:  {'train': 4703}
Dataloaders size:  1
Training classes:  ['cassetePlayer', 'chainShaw', 'church', 'englishSpringer', 'frenchHorn']
Training lengths:  5
Height x Width: 224 x 224
@@@@ STARTING TRAINING! @@@@
- Process 6 corresponds to GPU 0 of node 3
Rank:  6

Datasets size:  {'train': 4703}
Dataloaders size:  1
Training classes:  ['cassetePlayer', 'chainShaw', 'church', 'englishSpringer', 'frenchHorn']
Training lengths:  5
Height x Width: 224 x 224
@@@@ STARTING TRAINING! @@@@
- Process 3 corresponds to GPU 1 of node 1
Rank:  3

Datasets size:  {'train': 4703}
Dataloaders size:  1
Training classes:  ['cassetePlayer', 'chainShaw', 'church', 'englishSpringer', 'frenchHorn']
Training lengths:  5
Height x Width: 224 x 224
@@@@ STARTING TRAINING! @@@@
- Process 1 corresponds to GPU 1 of node 0
Rank:  1

Datasets size:  {'train': 4703}
Dataloaders size:  1
Training classes:  ['cassetePlayer', 'chainShaw', 'church', 'englishSpringer', 'frenchHorn']
Training lengths:  5
Height x Width: 224 x 224
@@@@ STARTING TRAINING! @@@@
- Process 5 corresponds to GPU 1 of node 2
Rank:  5

Datasets size:  {'train': 4703}
Dataloaders size:  1
Training classes:  ['cassetePlayer', 'chainShaw', 'church', 'englishSpringer', 'frenchHorn']
Training lengths:  5
Height x Width: 224 x 224
@@@@ STARTING TRAINING! @@@@
>>> Training on  4  nodes and  8  processes, master node is  sdumont3076
- Process 0 corresponds to GPU 0 of node 0
Rank:  0

Datasets size:  {'train': 4703}
Dataloaders size:  1
Training classes:  ['cassetePlayer', 'chainShaw', 'church', 'englishSpringer', 'frenchHorn']
Training lengths:  5
Height x Width: 224 x 224
@@@@ STARTING TRAINING! @@@@
[W NNPACK.cpp:80] Could not initialize NNPACK! Reason: Unsupported hardware.
[W NNPACK.cpp:80] Could not initialize NNPACK! Reason: Unsupported hardware.
[W NNPACK.cpp:80] Could not initialize NNPACK! Reason: Unsupported hardware.
[W NNPACK.cpp:80] Could not initialize NNPACK! Reason: Unsupported hardware.
[W NNPACK.cpp:80] Could not initialize NNPACK! Reason: Unsupported hardware.
[W NNPACK.cpp:80] Could not initialize NNPACK! Reason: Unsupported hardware.
[W NNPACK.cpp:80] Could not initialize NNPACK! Reason: Unsupported hardware.
[W NNPACK.cpp:80] Could not initialize NNPACK! Reason: Unsupported hardware.
Epoch [1/5], Step [20/294], Loss: 1.6294, Time data load: 0.236ms, Time training: 6.148ms
Epoch [1/5], Step [40/294], Loss: 1.2904, Time data load: 0.290ms, Time training: 43.442ms
Epoch [1/5], Step [60/294], Loss: 1.0279, Time data load: 0.295ms, Time training: 43.522ms
Epoch [1/5], Step [80/294], Loss: 1.2144, Time data load: 0.267ms, Time training: 43.470ms
Epoch [1/5], Step [100/294], Loss: 2.2554, Time data load: 0.371ms, Time training: 44.152ms
Epoch [1/5], Step [120/294], Loss: 1.4910, Time data load: 0.310ms, Time training: 43.286ms
Epoch [1/5], Step [140/294], Loss: 1.7565, Time data load: 0.324ms, Time training: 43.483ms
Epoch [1/5], Step [160/294], Loss: 0.5713, Time data load: 0.283ms, Time training: 43.549ms
Epoch [1/5], Step [180/294], Loss: 1.9261, Time data load: 0.257ms, Time training: 43.224ms
Epoch [1/5], Step [200/294], Loss: 1.6516, Time data load: 0.304ms, Time training: 43.839ms
Epoch [1/5], Step [220/294], Loss: 1.7962, Time data load: 0.287ms, Time training: 43.443ms
Epoch [1/5], Step [240/294], Loss: 1.7028, Time data load: 0.309ms, Time training: 43.437ms
Epoch [1/5], Step [260/294], Loss: 1.5078, Time data load: 0.330ms, Time training: 43.204ms
Epoch [1/5], Step [280/294], Loss: 0.9087, Time data load: 0.267ms, Time training: 43.506ms
------- Before checkpoint 1
------- After checkpoint 1
Epoch [2/5], Step [20/294], Loss: 1.1357, Time data load: 0.294ms, Time training: 43.558ms
Epoch [2/5], Step [40/294], Loss: 0.7266, Time data load: 0.284ms, Time training: 43.374ms
Epoch [2/5], Step [60/294], Loss: 1.1628, Time data load: 0.261ms, Time training: 43.523ms
Epoch [2/5], Step [80/294], Loss: 1.2303, Time data load: 0.292ms, Time training: 43.373ms
Epoch [2/5], Step [100/294], Loss: 2.1867, Time data load: 0.258ms, Time training: 43.489ms
Epoch [2/5], Step [120/294], Loss: 0.9078, Time data load: 0.259ms, Time training: 43.581ms
Epoch [2/5], Step [140/294], Loss: 1.6333, Time data load: 0.279ms, Time training: 43.287ms
Epoch [2/5], Step [160/294], Loss: 1.5204, Time data load: 0.298ms, Time training: 43.472ms
Epoch [2/5], Step [180/294], Loss: 1.9937, Time data load: 0.278ms, Time training: 43.412ms
Epoch [2/5], Step [200/294], Loss: 1.3296, Time data load: 0.284ms, Time training: 43.418ms
Epoch [2/5], Step [220/294], Loss: 1.4972, Time data load: 0.342ms, Time training: 43.376ms
Epoch [2/5], Step [240/294], Loss: 0.3823, Time data load: 0.260ms, Time training: 43.440ms
Epoch [2/5], Step [260/294], Loss: 1.4697, Time data load: 0.277ms, Time training: 43.467ms
Epoch [2/5], Step [280/294], Loss: 1.9646, Time data load: 0.287ms, Time training: 43.506ms
------- Before checkpoint 2
------- After checkpoint 2
Epoch [3/5], Step [20/294], Loss: 0.3786, Time data load: 0.261ms, Time training: 43.505ms
Epoch [3/5], Step [40/294], Loss: 0.9506, Time data load: 0.257ms, Time training: 43.456ms
Epoch [3/5], Step [60/294], Loss: 0.9734, Time data load: 0.269ms, Time training: 43.478ms
Epoch [3/5], Step [80/294], Loss: 1.7417, Time data load: 0.253ms, Time training: 43.501ms
Epoch [3/5], Step [100/294], Loss: 1.0803, Time data load: 0.471ms, Time training: 43.156ms
Epoch [3/5], Step [120/294], Loss: 1.1047, Time data load: 0.291ms, Time training: 43.444ms
Epoch [3/5], Step [140/294], Loss: 1.4598, Time data load: 0.252ms, Time training: 43.364ms
Epoch [3/5], Step [160/294], Loss: 1.2450, Time data load: 0.265ms, Time training: 43.421ms
Epoch [3/5], Step [180/294], Loss: 3.0937, Time data load: 0.251ms, Time training: 43.546ms
Epoch [3/5], Step [200/294], Loss: 0.9215, Time data load: 0.253ms, Time training: 43.568ms
Epoch [3/5], Step [220/294], Loss: 1.7584, Time data load: 0.281ms, Time training: 43.317ms
Epoch [3/5], Step [240/294], Loss: 1.3332, Time data load: 0.281ms, Time training: 43.362ms
Epoch [3/5], Step [260/294], Loss: 1.4534, Time data load: 0.274ms, Time training: 43.375ms
Epoch [3/5], Step [280/294], Loss: 0.9775, Time data load: 0.284ms, Time training: 43.522ms
------- Before checkpoint 3
------- After checkpoint 3
Epoch [4/5], Step [20/294], Loss: 0.6847, Time data load: 0.273ms, Time training: 6.175ms
Epoch [4/5], Step [40/294], Loss: 0.5368, Time data load: 0.339ms, Time training: 43.419ms
Epoch [4/5], Step [60/294], Loss: 0.8154, Time data load: 0.295ms, Time training: 43.449ms
Epoch [4/5], Step [80/294], Loss: 2.3641, Time data load: 0.767ms, Time training: 42.740ms
Epoch [4/5], Step [100/294], Loss: 1.2218, Time data load: 0.258ms, Time training: 43.437ms
Epoch [4/5], Step [120/294], Loss: 1.2024, Time data load: 0.237ms, Time training: 5.920ms
Epoch [4/5], Step [140/294], Loss: 1.1091, Time data load: 0.241ms, Time training: 43.568ms
Epoch [4/5], Step [160/294], Loss: 0.2790, Time data load: 0.269ms, Time training: 43.473ms
Epoch [4/5], Step [180/294], Loss: 2.8331, Time data load: 0.251ms, Time training: 43.435ms
Epoch [4/5], Step [200/294], Loss: 0.9092, Time data load: 0.257ms, Time training: 43.386ms
Epoch [4/5], Step [220/294], Loss: 1.3395, Time data load: 0.268ms, Time training: 5.990ms
Epoch [4/5], Step [240/294], Loss: 0.8491, Time data load: 0.250ms, Time training: 43.603ms
Epoch [4/5], Step [260/294], Loss: 1.9812, Time data load: 0.247ms, Time training: 43.321ms
Epoch [4/5], Step [280/294], Loss: 1.6657, Time data load: 0.271ms, Time training: 43.549ms
------- Before checkpoint 4
------- After checkpoint 4
Epoch [5/5], Step [20/294], Loss: 0.4346, Time data load: 0.270ms, Time training: 43.382ms
Epoch [5/5], Step [40/294], Loss: 1.2120, Time data load: 0.292ms, Time training: 43.391ms
Epoch [5/5], Step [60/294], Loss: 1.2257, Time data load: 0.273ms, Time training: 43.528ms
Epoch [5/5], Step [80/294], Loss: 1.6547, Time data load: 0.456ms, Time training: 43.300ms
Epoch [5/5], Step [100/294], Loss: 1.3733, Time data load: 0.237ms, Time training: 43.736ms
Epoch [5/5], Step [120/294], Loss: 1.3180, Time data load: 0.257ms, Time training: 43.510ms
Epoch [5/5], Step [140/294], Loss: 0.6443, Time data load: 0.300ms, Time training: 43.551ms
Epoch [5/5], Step [160/294], Loss: 0.7339, Time data load: 0.250ms, Time training: 43.416ms
Epoch [5/5], Step [180/294], Loss: 0.8735, Time data load: 0.257ms, Time training: 43.540ms
Epoch [5/5], Step [200/294], Loss: 1.1292, Time data load: 0.284ms, Time training: 43.496ms
Epoch [5/5], Step [220/294], Loss: 1.6093, Time data load: 0.258ms, Time training: 43.520ms
Epoch [5/5], Step [240/294], Loss: 0.3755, Time data load: 0.286ms, Time training: 43.526ms
Epoch [5/5], Step [260/294], Loss: 0.8125, Time data load: 0.289ms, Time training: 43.514ms
Epoch [5/5], Step [280/294], Loss: 2.1883, Time data load: 0.287ms, Time training: 43.487ms
@@@@ END TRAINING - Rank: 0! @@@@
>>> Training complete in: 0:01:07.573225
@@@@ END TRAINING - Rank: 1! @@@@
@@@@ END TRAINING - Rank: 7! @@@@
@@@@ END TRAINING - Rank: 6! @@@@
@@@@ END TRAINING - Rank: 5! @@@@
@@@@ END TRAINING - Rank: 4! @@@@
@@@@ END TRAINING - Rank: 3! @@@@
@@@@ END TRAINING - Rank: 2! @@@@
-- end --------------------------------
<2. quit>                    Mon Apr  4 16:00:33 -03 2022
```





## Author

[**Valdivino Alexandre de Santiago J&uacute;nior**](https://www.linkedin.com/in/valdivino-alexandre-de-santiago-j%C3%BAnior-103109206/?locale=en_US)

## Licence

This project is licensed under the GNU GENERAL PUBLIC LICENSE, Version 3 (GPLv3) - see the [LICENSE.md](../LICENSE) file for details.

## Cite

Please cite this repository if you use it as:

V. A. Santiago J&uacute;nior. Project IDeepS, 2022. Acessed on: *date of access*. Available: https://github.com/vsantjr/IDeepS. 