import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import Compose
import os
import argparse
from utility import *
from baseConfig import Engine, options, make_dataset
import time

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Hyperspectral Image Denoising')
    opt = options(parser)
    print("tesing--------------------------------------------------------------------------------------------------------")
    print("------------------------------"+opt.arch)
    print("------------------------------"+opt.testdir)
    
    engine = Engine(opt)
    basefolder = opt.testdir
    
    mat_datasets = [MatDataFromFolder(
        basefolder) ]
    
    if opt.trainDataset == 'urban': 
        mat_datasets = [MatDataFromFolder(
        opt.testDir, size=opt.test_matSize,fns=['Urban_304.mat'])]


    if not engine.get_net().use_2dconv:
        mat_transform = Compose([
            LoadMatHSI(input_key='input', gt_key='gt',
                    transform=lambda x:x[ ...][None], needsigma=False),
        ])
    else:
        mat_transform = Compose([
            LoadMatHSI(input_key='input', gt_key='gt', needsigma=False),
        ])

    mat_datasets = [TransformDataset(mat_dataset, mat_transform)
                    for mat_dataset in mat_datasets]

 
    mat_loaders = [DataLoader(
        mat_dataset,
        batch_size=1, shuffle=False,
        num_workers=1, pin_memory=opt.no_cuda
        
    ) for mat_dataset in mat_datasets]        

    print(mat_loaders[0])

    strart_time = time.time()
    # engine.judge(mat_loaders[0], basefolder)
    engine.test(mat_loaders[0], basefolder)
    end_time = time.time()
    test_time = end_time-strart_time
    print('cost-time: ',(test_time/50))
