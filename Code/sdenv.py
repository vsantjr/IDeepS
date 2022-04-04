'''
Authors: Eduardo Furlan Miranda and Valdivino Alexandre de Santiago Júnior
This program was developed based on recommendations from IDRIS (http://www.idris.fr/eng/jean-zay/gpu/jean-zay-gpu-torch-multi-eng.html). Its goal is to expose the Slurm parameter values according to the subscription script (.srm) in order to run jobs in multiple GPUs. Import this file into your Python code.
'''

import os, hostlist

job_partition = os.uname()[1]
hostnames = [job_partition]

if 'SLURM_PROCID' in os.environ:
    # Get SLURM variables:

    # Equivalent to MPI rank
    rank          = int(os.environ['SLURM_PROCID'])

    # rank inside a node
    local_rank    = int(os.environ['SLURM_LOCALID'])

    # node rank inside NODELIST
    node_rank     = int(os.environ['SLURM_NODEID'])

    size          = int(os.environ['SLURM_NTASKS'])
    cpus_per_task = int(os.environ['SLURM_CPUS_PER_TASK'])
    job_partition = os.environ['SLURM_JOB_PARTITION']
    hostnames = hostlist.expand_hostlist(os.environ['SLURM_JOB_NODELIST'])

# Get IDs of GPUs according to the type of partition/queue.
gpu_ids = []
if job_partition in ['nvidia', 'nvidia_small', 'nvidia_dev', 'nvidia_scal', 'nvidia_long', 'het_scal']:
        gpu_ids = [0, 1]
elif job_partition in ['sequana_gpu_shared', 'sdumont18']:
        gpu_ids = [0, 1, 2, 3]
elif job_partition in ['gdl']:
        gpu_ids = [0]

# Define the MASTER.
MASTER_ADDR = hostnames[0]
os.environ['MASTER_ADDR'] = MASTER_ADDR

# To avoid port conflict on the same node.
MASTER_PORT = str(9000 + int(min(gpu_ids)))
os.environ['MASTER_PORT'] = MASTER_PORT
