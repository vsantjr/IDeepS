module load cuda/11.1
module load cudnn/8.2_cuda-11.1
module load gcc/8.3

conda install astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
conda install -c pytorch magma-cuda111

git clone --recursive https://github.com/pytorch/pytorch
cd pytorch

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py install

git clone --recursive https://github.com/pytorch/vision
cd vision
python setup.py install
