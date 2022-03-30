# Project IDeepS

This repository is related to the project ***Classificação de imagens via redes neurais profundas e grandes bases de dados para aplicações aeroespaciais*** (Image classification via Deep neural networks and large databases for aeroSpace applications - IDeepS). The IDeepS project is supported by the *Laboratório Nacional de Computação Científica* (LNCC/MCTI, Brazil) via resources of the [SDumont](http://sdumont.lncc.br) supercomputer.

The main goal of the repository is to provide directives on how you can perform the setup and run deep learning (DL) applications in the SDumont supercomputer. We consider the DL framework [PyTorch](https://pytorch.org/) to run the DL code.


## Overview of the SDumont Supercomputer

The SDumont supercomputer has an installed processing capacity of 5.1 Petaflop/s presenting a hybrid configuration of computational nodes, in terms of the available parallel processing architecture. Currently, SDumont has a total of 36,472 CPU cores distributed across 1,134 computing nodes, most of which are made up exclusively of CPUs with a multi-core architecture. There is, however, a significant amount of additional nodes that, in addition to the same multi-core CPUs, contain device types with the so-called many-core architecture: GPU and MIC.

There are several [node configurations](https://sdumont.lncc.br/machine.php?pg=machine#) but here we show only the nodes the IDeepS project uses:

- **B715**. 198 B715 computing nodes (thin node) where each node has 2 x Intel Xeon E5-2695v2 Ivy Bridge CPU, 2 x NVIDIA K40 GPUs, and 64 GB of RAM;
- **AI**. 1 artificial intelligence (AI) node with 2 x Intel Xeon Skylake Gold 6148, 8 x NVIDIA Tesla V100 GPUs with NVLink, and 384 GB of RAM;
- **BSeq**. 94 Bull Sequana X1120 computing nodes where each has 2 x Intel Xeon Skylake 6252 CPU, 4 x NVIDIA Volta V100 GPUs, and 384 GB of RAM.


## Jobs Queues

SDumont's operating system is [RedHat Linux 7.6](https://www.redhat.com/pt-br). Job submission must be done via the cluster management and job scheduling system [Slurm](https://slurm.schedmd.com/documentation.html). Below we show the type of jobs queues the IDeepS project can use. The maximum wallclock refers to the maximum time a job can run in the respective node without being interrupted by a timeout.


| Queue Name  	| Node 			| Maximum Wallclock (h)	|
| ------------- | ------------- |----------------------	|
| nvidia  		| B715  		|48 (2 days) 					|
| nvidia_small  		| B715  		|1					|
| nvidia_dev  		| B715  		|00:20 					|
| nvidia_scal  		| B715  		|18					|
| nvidia_long  		| B715  		|744 (31 days) 					|
| het_scal 		| B715  		|18					|
| gdl 		| AI 		|48 (2 days)					|
| sequana_gpu_shared  		| BSeq  		|96 (4 days) 					|




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

After installing Miniconda, create a new conda environment with a selected version of Python. For instance, if the selected Python version is 3.9.6, then we create the ```myenv``` environment in this way:

```
conda create -n myenv python=3.9.6
```

**IMPORTANT**: It is recommended to create as many different conda environments as necessary, specially if you want to work with several DL frameworks/libraries. For instance, if you need to work not only with PyTorch but also with [TensorFlow](https://www.tensorflow.org/), thus create a new conda environment to submit jobs with TensorFlow. This avoid potential conflicts. Moreover, some available DL models were developed and only work with specific versions of the DL libraries/frameworks. Thus, creating new conda environments for such versions is suggested.

#### PyTorch

It is required to build PyTorch from source since its newer versions do not support the NVIDIA K40 GPUs. If you install the PyTorch in the ordinary way (via conda, pip), you will be allowed to submit jobs in only two out of the queues presented above: gdl and sequana_gpu_shared. 

Firstly, activate your newly created environment:

```
conda activate myenv
```

[Here](./Utils/pytorchsrc.sh), we present a shell script that installs PyTorch from source. Hence, in the scratch dir (```/scratch/proj/name.user```), run:

```
bash pytorchsrc.sh
```

Note that such installation can take a considerable time to complete.








## Author

[**Valdivino Alexandre de Santiago J&uacute;nior**](https://www.linkedin.com/in/valdivino-alexandre-de-santiago-j%C3%BAnior-103109206/?locale=en_US)

## Licence

This project is licensed under the GNU GENERAL PUBLIC LICENSE, Version 3 (GPLv3) - see the [LICENSE.md](LICENSE) file for details.

## Cite

Please cite this repository if you use it as:

V. A. Santiago J&uacute;nior. Project IDeepS, 2022. Acessed on: *date of access*. Available: https://github.com/vsantjr/IDeepS. 