'''
Authors: Valdivino Alexandre de Santiago Júnior and Eduardo Furlan Miranda
This program was developed based on recommendations from IDRIS (http://www.idris.fr/eng/jean-zay/gpu/jean-zay-gpu-torch-multi-eng.html). It is a program that shows how to distribute a convolutional neural network (CNN) model implemented PyTorch. Via SLURM, it allows to use multiple GPUs in a single node or in multiple nodes.
'''

'''
Before running the code, do the following:
1.) Download the imgnet320_c5.zip (https://github.com/vsantjr/IDeepS/blob/master/Datasets/test.zip) dataset which is a subset of the imagenettetvt320 (https://www.kaggle.com/datasets/valdivinosantiago/imagenettetvt320) dataset with only 5 classes, training and test datasets. But here, we will only use the training dataset within the training phase;
2.) Create a directory in your working directory called "img";
3.) Unzip the dataset into the "img" directory: unzip imgnet320_c5.zip -d ./img. 
'''

import os
from datetime import datetime
from time import time
import argparse
import torch.multiprocessing as mp
import torchvision
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.autograd import Variable
import sdenv
import numpy as np
import sys
import torch.nn.functional as F

# Important hyper-parameters/variables.
ninpf = 106 # Number of input features for the output (fully-connected) layer of the CNN.
nc = 5 # Number of classes.

# Custom dataset that includes image file paths. Extends torchvision.datasets.ImageFolder.
class FileNames(datasets.ImageFolder):
    
    # Override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # This is what ImageFolder normally returns 
        original_tuple = super(FileNames, self).__getitem__(index)
        # The image file path
        path = self.imgs[index][0]
        # Make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

# Main function.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size', default=128, type =int,
        help='batch size. it will be divided in mini-batch for each worker')
    parser.add_argument('-e','--epochs', default=2, type=int, metavar='N',
        help='number of total epochs to run')
    parser.add_argument('-c','--checkpoint', default=None, type=str,
        help='path to checkpoint to load')
    args = parser.parse_args()

    # Only the training phase here dealing only with the training dataset. But it is easy to work with other sets (e.g. test).
    train(args)   

# Destroy process group.
def cleanup():
    dist.destroy_process_group()

# A simple CNN.
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(24)
        self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(24)
        self.fc1 = nn.Linear(24*ninpf*ninpf, nc)

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))      
        output = F.relu(self.bn2(self.conv2(output)))     
        output = self.pool(output)                        
        output = F.relu(self.bn4(self.conv4(output)))     
        output = F.relu(self.bn5(self.conv5(output)))     
        output = output.view(-1, 24*ninpf*ninpf)
        output = self.fc1(output)

        return output

# The training phase.
def train(args):
    
    # The dataset image directory.
    data_dir = './img'
        
    # Images will be cropped to 224 x 224 
    hei_wid = 224 
        
    # Configure distribution method: define address and port of the master node and initialise communication backend (NCCL).
    print('Rank: ', sdenv.rank)
    dist.init_process_group(
        backend='nccl', 
        init_method='env://', 
        world_size=sdenv.size, 
        rank=sdenv.rank)
       
    # Distribute the model.
    torch.cuda.set_device(sdenv.local_rank)
    gpu = torch.device("cuda")
    model = SimpleCNN().to(gpu)
    ddp_model = DistributedDataParallel(
        model, 
        device_ids=[sdenv.local_rank])
    if args.checkpoint is not None:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % sdenv.local_rank}
        ddp_model.load_state_dict(
            torch.load(args.checkpoint, map_location=map_location))
    
    # Distribute batch size (mini-batch).
    batch_size = args.batch_size 
    batch_size_per_gpu = batch_size // sdenv.size
    
    # Transformations and data loading. 
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(hei_wid),
            transforms.RandomHorizontalFlip(),                         
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    image_datasets = {x: FileNames(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train']}

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        image_datasets['train'],
        num_replicas=sdenv.size,
        rank=sdenv.rank)
        
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size_per_gpu,shuffle=False, num_workers=4, pin_memory=True,sampler=train_sampler)
            for x in ['train']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}
    
    # Just checking the sizes. 
    print('\nDatasets size: ', dataset_sizes)
    print('Dataloaders size: ', len(dataloaders))
    class_names = image_datasets['train'].classes
    print('Training classes: ', class_names)
    print('Training lengths: ', len(class_names))
    print('Height x Width: {} x {}'.format(hei_wid,hei_wid))
    
    # Define loss function (criterion) and optimiser.
    criterion = nn.CrossEntropyLoss()  
    optimizer = torch.optim.SGD(ddp_model.parameters(), 1e-4)
      
    # Training (timers and display handled by process 0)
    if sdenv.rank == 0: 
        # Create checkpoint directory if it does not exist.
        if not os.path.exists('cptdir'):
            os.makedirs('cptdir')
        start = datetime.now()
    total_step = len(dataloaders['train'])
    
    print('@@@@ STARTING TRAINING! @@@@')
    for epoch in range(args.epochs):

        if sdenv.rank == 0: start_dataload = time()
        phase = 'train' # We have only the training phase here.
                
        for i, (images, labels, paths) in enumerate(dataloaders[phase]):     
            
            # Distribution of images and labels to all GPUs.
            images = images.to(gpu, non_blocking=True)
            labels = labels.to(gpu, non_blocking=True) 
            
            if sdenv.rank == 0: stop_dataload = time()
            if sdenv.rank == 0: start_training = time()
            
            # Forward pass.
            outputs = ddp_model(images)
            loss = criterion(outputs, labels)

            # Backward and optimise.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                                   
            if sdenv.rank == 0: 
                stop_training = time() 
            if (i + 1) % 20 == 0 and sdenv.rank == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Time data load: {:.3f}ms, Time training: {:.3f}ms'                      .format(epoch + 1, 
                              args.epochs, 
                              i + 1, 
                              total_step, 
                              loss.item(), 
                              (stop_dataload - start_dataload)*1000,
                              (stop_training - start_training)*1000))
            if sdenv.rank == 0: start_dataload = time()
        
    
        # Save checkpoint at every end of epoch except the last one.
        if (sdenv.rank == 0) and ((epoch+1) < args.epochs):
            print('------- Before checkpoint {}'.format(epoch+1))
            torch.save(
                ddp_model.state_dict(), 
                './cptdir/{}GPU_{}epoch.checkpoint'.format(sdenv.size, epoch+1))
            print('------- After checkpoint {}'.format(epoch+1))
        
    print('@@@@ END TRAINING - Rank: {}! @@@@'.format( sdenv.rank))
    
    cleanup()
    if sdenv.rank == 0:
        print(">>> Training complete in: "+str(datetime.now()-start))

# The main code.
if __name__ == '__main__':   
    # Display information.
    if sdenv.rank == 0:
        print(">>> Training on ", len(sdenv.hostnames), 
              " nodes and ", sdenv.size, 
              " processes, master node is ", sdenv.MASTER_ADDR)
    print("- Process {} corresponds to GPU {} of node {}"          .format(sdenv.rank, 
          sdenv.local_rank, 
          sdenv.node_rank))

    main()
