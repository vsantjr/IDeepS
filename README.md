# Deep Learning Made Easy

This repository provides examples for those who want to start working in deep learning (DL), a subfield of machine learning (ML). 


## Convolutional Neural Networks (CNNs)

The following CNNs are available:


- PyTorch\_MNIST: Google Colab's notebook for the handwritten digit classification problem based on the classical Modified National Institute of Standards and Technology (MNIST) database. It uses three neural networks to address this problem: **SNN500** by [Aviv Shamsian](https://github.com/AvivSham/Pytorch-MNIST-colab), **CNN3L** by [Nutan](https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118), and **LeNet-5** by [Bolla Karthikeya](https://github.com/bollakarthikeya/LeNet-5-PyTorch/blob/master/lenet5\_gpu.py);
- PyTorch\_MNIST\_Profiler: Google Colab's notebook related to the handwritten digit classification problem based on the classical MNIST database. Hovewer, its goal is to address performance bottlenecks of the network/model via the TensorBoard Plugin with PyTorch Profiler. Hence, it will not cover the classification task completely. It uses the same three neural networks of the PyTorch\_MNIST notebook; 
- PyTorch\_CIFAR-10: Google Colab's notebook for the image classification problem based on the classical CIFAR-10 database. It is a modification of the [*Training a Classifier*](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) PyTorch tutorial. It uses two neural networks to address this problem: **CNN3L** by [Nutan](https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118) and **LeNet-5** presented in the *Training a Classifier* tutorial;
- PyTorch\_CIFAR-10\_Profiler: Google Colab's notebook related to the image classification problem based on the classical CIFAR-10 database. Hovewer, its goal is to address performance bottlenecks of the network/model via the TensorBoard Plugin with PyTorch Profiler. Hence, it will not cover the classification task completely. It uses same two neural networks of the PyTorch\_CIFAR-10 notebook;
- PyTorch\_VGG\_ResNet\_VGGout: It shows how easy is to create "new" deep neural networks (DNNs) by changing a few lines of code of previously proposed models. New networks: VGG12BN, ResNet-14. The output is due to the execution of VGG12BN;
- PyTorch\_VGG\_ResNet\_ResNetout: Same previous notebook but with the output of ResNet-14;
- PyTorch\_DenseNet: It shows how easy is to create "new" deep neural networks (DNNs) by changing a few lines of code of previously proposed models. New network: DenseNet-83;
- PyTorch\_TransferLayer: This notebook shows how to use transfer learning (TL) within a supervised context. The TL technique is fine-tuning the deep convolutional neural network (CNN). We considered six pretrained models: ResNet-18, ResNet-34, DenseNet-121, DenseNet-161, GoogLeNet, and Inception-v3;
- PyTorch\_U-Net: This notebook is about semantic segmentation via U-Net. **Important**. This notebook was developed by the [Albumentations Team](https://albumentations.ai/) and a few modifications have been done by Valdivino Alexandre de Santiago J&uacute;nior. If you want to use Albumentations in your study, please cite their [article](https://www.mdpi.com/2078-2489/11/2/125);
- PyTorch\_Mask R-CNN: This notebook is about instance segmentation via Mask R-CNN. **Important**. This notebook was developed by [Erdene-Ochir Tuguldur](https://github.com/tugstugi/dl-colab-notebooks/blob/master/notebooks/TorchvisionMaskRCNN.ipynb) and a few modifications have been done by Valdivino Alexandre de Santiago Júnior.


## Generative Adversarial Networks (GANs)

The following GANs are available:

- PyTorch\_DCGAN: This notebook addresses the [deep convolutional generative adversarial network](https://arxiv.org/abs/1511.06434) (DCGAN);
- PyTorch\_PROGAN: This notebook is about the [progressive growing of GANs](https://arxiv.org/abs/1710.10196) (PROGAN);
- PyTorch\_CGAN: This notebook is about the [conditional generative adversarial network](https://arxiv.org/pdf/1411.1784.pdf) (CGAN).



## Datasets

The following datasets are required to download into Google Drive so that some notebooks can work properly:

- [imagenettetvt320](https://www.kaggle.com/valdivinosantiago/imagenettetvt320);
- [CelebFaces Attributes](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).


## Author

[**Valdivino Alexandre de Santiago J&uacute;nior**](https://www.linkedin.com/in/valdivino-alexandre-de-santiago-j%C3%BAnior-103109206/?locale=en_US)

## Licence

This project is licensed under the GNU GENERAL PUBLIC LICENSE, Version 3 (GPLv3) - see the [LICENSE.md](LICENSE) file for details.

## Cite

Please cite this repository if you use it as:

V. A. Santiago J&uacute;nior. Deep Leaning Made Easy, 2021. Acessed on: *date of access*. Available: https://github.com/vsantjr/DeepLearningMadeEasy. 


