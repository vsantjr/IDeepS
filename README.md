# Project IDeepS


<img src="https://github.com/vsantjr/CAP/blob/master/Images/logo1ideeps.png" width=58%>

This repository is related to the project ***Classificação de imagens via redes neurais profundas e grandes bases de dados para aplicações aeroespaciais*** (Image classification via Deep neural networks and large databases for aeroSpace applications - IDeepS). The IDeepS project is supported by the *Laboratório Nacional de Computação Científica* ([LNCC](https://www.gov.br/lncc/pt-br)/MCTI, Brazil) via resources of the [SDumont](http://sdumont.lncc.br) supercomputer.

The main goal of the repository is to provide directives/suggestions on how you can perform the setup and run deep learning (DL) applications in the SDumont supercomputer. We mainly consider the DL framework [PyTorch](https://pytorch.org/) to run the DL code but there are some remarks to use the library [TensorFlow](https://www.tensorflow.org/). The suggestions below are based on a Standard-type project which is the category of project IDeepS belongs.

Researchers, professors and post-graduate students from the following organisations are involved in the project: *Instituto Nacional de Pesquisas Espaciais* [(INPE)](https://www.gov.br/inpe/pt-br), *Instituto de Estudos Avançados* [(IEAv)](https://ieav.dcta.mil.br/), *Universidade Federal de São Paulo - Campus São José dos Campos* [(UNIFESP)](https://www.unifesp.br/campus/sjc/), *Instituto Tecnológico de Aeronáutica* [(ITA)](http://www.ita.br/), and *Universidade Federal de São Carlos - Campus Sorocaba* [(UFSCar)](https://www.sorocaba.ufscar.br/).


## Overview of the SDumont Supercomputer

The SDumont supercomputer has an installed processing capacity of 5.1 Petaflop/s presenting a hybrid configuration of computational nodes, in terms of the available parallel processing architecture. Currently, SDumont has a total of 36,472 CPU cores distributed across 1,134 computing nodes, most of which are made up exclusively of CPUs with a multi-core architecture. There is, however, a significant amount of additional nodes that, in addition to the same multi-core CPUs, contain device types with the so-called many-core architecture: GPU and MIC. More detailed information about the SDumont supercomputer can be seen [here](https://sdumont.lncc.br/support_manual.php?pg=support#1).

There are several node configurations but here we show only the nodes the IDeepS project uses:

- **B715**. 198 B715 computing nodes (thin node) where each node has 2 x Intel Xeon E5-2695v2 Ivy Bridge CPU, 2 x NVIDIA K40 GPUs, and 64 GB of RAM;
- **AI**. 1 artificial intelligence (AI) node with 2 x Intel Xeon Skylake Gold 6148, 8 x NVIDIA Tesla V100 GPUs with NVLink, and 384 GB of RAM;
- **BSeq**. 94 Bull Sequana X1120 computing nodes where each has 2 x Intel Xeon Skylake 6252 CPU, 4 x NVIDIA Volta V100 GPUs, and 384 GB of RAM.


## Jobs Queues

SDumont's operating system is [RedHat Linux 7.6](https://www.redhat.com/pt-br). Job submission must be done via the cluster management and job scheduling system [Slurm](https://slurm.schedmd.com/documentation.html). Below we show the type of jobs queues the IDeepS project can use. In the Node column, we show the type of node related to the queue but note that other nodes may be used in some queues. For instance, the het_scal queue has 64 nodes with CPU architecture and 64 nodes with GPU architecture. The jobs submitted to
this queue can request a set of heterogeneous nodes for execution. But, in the case of the IDeepS project, the most desired resources are the nodes with GPU due to the characteristics of DL applications. The Maximum Wallclock refers to the maximum time a job can run in the respective node without being interrupted by a timeout.


| Queue Name  	| Node 			| GPUs/Node   | Maximum Wallclock (h)	|
| ------------- | ------------- |-------------|-------------------------|
| nvidia  		| B715  		|2            |48 (2 days) 				|
| nvidia_small  		| B715  		|2    |1					|
| nvidia_dev  		| B715  		|2        |00:20 					|
| nvidia_scal  		| B715  		|2        |18					|
| nvidia_long  		| B715  		|2        |744 (31 days) 		|
| het_scal 		| B715  		|2            |18					|
| gdl 		| AI 		|8                    |48 (2 days)				|
| sequana_gpu_shared  		| BSeq  	|4    |96 (4 days) 				|




## Connection, Login and File Transference

In order to use the SDumont supercomputer, you should connect to the LNCC's VPN using the login and password provided to you. After connecting, you must login using the SSH network protocol. File transference from and to SDumont can be done via the SCP network protocol.

[Here](./Mark/utsh.md), we present information and a simple shell script that can help to login and transfer files from/to SDumont.

**IMPORTANT**: It is likely you will experience broken pipe connection errors when using SSH. Hence, change your ```ssh_config``` (client) file so that it resembles something like this:

```
Host *
        ServerAliveInterval 30
        IPQoS=throughput
```

In MacOS, this file is usually inside directory ```/etc/ssh```. The ```ServerAliveInterval``` option specifies the number of seconds between keepalives. In other words, this is a timeout interval, in seconds, after which if no data has been received from the server, SSH will send a message through the encrypted channel to request a response from the server.


## Installing Software

When in your SDumont account, change to your ```scratch``` directory. This directory is used to store all the files that will be used during the execution of a job (submission scripts, executables, input data, output data, etc).

```cd $SCRATCH```

Running a pwd command, you will see something like this (let us assume that the name of the user is **name.user** and the name of the project is **proj**):

```/scratch/proj/name.user```

#### Miniconda

The first software that it is interesting to install is [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for Linux. Hence, download the appropriate shell file (```Miniconda3....sh```) and run:

```
bash Miniconda3....sh
```

You may also prefer using one of the Anaconda versions already installed in SDumont rather than installing Miniconda. After installing Miniconda, create a new conda environment with a selected version of Python. But firstly, deactivate the default conda environment (```base```), assuming that this environment is currently activated:

```
conda deactivate
```

Thus, if the selected Python version is 3.9.6, then create the ```myenv``` environment in this way:

```
conda create -n myenv python=3.9.6
```


#### PyTorch

It is required to build PyTorch from source since its newer versions do not support the NVIDIA K40 GPUs. If you install PyTorch in the ordinary way (via conda, pip), you will be allowed to submit jobs in only two out of the queues presented earlier: gdl and sequana_gpu_shared. 

Firstly, activate your newly created environment:

```
conda activate myenv
```

[Here](./Utils/pytorchsrc.sh), we present a shell script that installs PyTorch from source. Hence, in the scratch directory (```/scratch/proj/name.user```), run:

```
bash pytorchsrc.sh
```

Note that such installation can take a considerable time to complete.

**IMPORTANT**: If you need to install some libraries, packages, etc. it is highly recommended to do it via conda within the ennvironment you need such libraries, packages. Conda is interesting because of its clear structure, transparent file management (no installation of files outside its directory), lots of available packages, and so on. However, bear in mind that, depending on the software you want to install, you might eventually ask permission from LNCC.

#### TensorFlow

If you want to use TensorFlow rather than or in addition to PyTorch, at least for now, it is not necessary to install TensorFlow from source as it is the case for PyTorch. However, it is highly recommended to create as many different conda environments as necessary, if you want to work with several DL frameworks/libraries. Thus, create a new conda environment to submit jobs with TensorFlow. This avoid potential conflicts. Moreover, some available DL models were developed and only work with specific versions of the DL libraries/frameworks. Thus, creating new conda environments for such versions is suggested.

Hence, you can create a new environment (```myenvtf```) and install [TensorFlow](https://docs.anaconda.com/anaconda/user-guide/tasks/tensorflow/) as shown below:

```
conda create -n myenvtf tensorflow-gpu
```

After that, just activate the environment to use TensorFlow:

```
conda activate myenvtf
```


## Submitting Jobs

In order to submit a job, basically you need to follow the steps below:

- As previously mentioned, all the necessary files (executable, libraries, input data) must be in the ```scratch``` directory; 
- Create a submission script (```.srm```), configuring the parameters necessary for the execution of the job;
- Submit the script (```.srm```) with the command ```sbatch```. Thus, if your script is ```test.srm```, you run in the terminal: ```sbatch test.srm```;
- If you want to see the outputs of your application during its execution, you can run in the terminal: ```cp slurm-ID.out a.txt```, where ID is the job ID (number) provided by Slurm. The outputs of your application are redirected to this ```.out``` file. Hence, you may just call ```vim a.txt``` to see the current snapshot of your execution. 

Moreover, Slurm provides several useful user commands, in addition to ```sbatch```, to work with a supercomputer. From these, it is worth mentioning: ```squeue```, ```sacct```, ```scancel```, and ```srun```. See more details [here](https://slurm.schedmd.com/quickstart.html#:~:text=The%20slurmd%20daemons%20provide%20fault,run%20anywhere%20in%20the%20cluster.) and [here](https://sdumont.lncc.br/support_manual.php?pg=support#7).

For instance, if you want to cancel a submitted job, you type (ID is the job ID):

```
scancel ID
```

Moreover, let us say that you want to see what is the current status of the sequana_gpu_shared queue to where you have just submitted a job, then you type:

```
squeue | grep sequana_g
```


After executing the command ```squeue``` or ```sacct``` , the job can be in one of the states presented [here](https://slurm.schedmd.com/squeue.html) and also [here](https://sdumont.lncc.br/support_manual.php?pg=support#7).


In a DL project like IDeepS, we can divide the job submission in three configurations (click on the respective links to see examples of submission scripts and code): [single node/single GPU](https://github.com/vsantjr/IDeepS/blob/master/Mark/sinnode.md#single-nodesingle-gpu), [single node/multiple GPUs](https://github.com/vsantjr/IDeepS/blob/master/Mark/sinnodemg.md#single-nodemultiple-gpus), and [multiple nodes/multiple GPUs](./Mark/mulnode_mulgpu.md).

## JupyterLab

JupyteLab can also be executed in SDumont to support interactive computing. Click [here](https://sites.usp.br/cadase/recursos-computacionais/tutoriais-sdumont/) to see how you can start JupyterLab in SDumont and access your notebooks stored in the supercomputer. 

However, for more non-trivial DL applications, the most suitable approach to take advantages of a supercomputer seems to be as shown in the Submitting Jobs subsections above, by creating a separate submission script (```.srm```) and calling Python within such a script.

But you can start JupyterLab locally in your machine and launch a terminal session as shown in the figure below.

<img src="https://github.com/vsantjr/CAP/blob/master/Images/jl1.png" width=90%>

You can then use the previous [script](./Utils/ut.sh) to login and transfer files from/to SDumont.

<img src="https://github.com/vsantjr/CAP/blob/master/Images/jl2.png" width=90%>

You can even work with notebooks locally, export the executable script (```.py```) to your machine, and transfer the programs to SDumont.

<img src="https://github.com/vsantjr/CAP/blob/master/Images/jl3.png" width=90%>






## Author

[**Valdivino Alexandre de Santiago J&uacute;nior**](https://www.linkedin.com/in/valdivino-alexandre-de-santiago-j%C3%BAnior-103109206/?locale=en_US)

## Licence

This project is licensed under the GNU GENERAL PUBLIC LICENSE, Version 3 (GPLv3) - see the [LICENSE.md](LICENSE) file for details.

## Cite

Please cite this repository if you use it as:

V. A. Santiago J&uacute;nior. Project IDeepS, 2022. Acessed on: *date of access*. Available: https://github.com/vsantjr/IDeepS. 