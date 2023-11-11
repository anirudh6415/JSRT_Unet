import numpy as np
import os
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from misc import multiclass_masks
from PIL import Image
import cv2



class segmentation_dataset(Dataset):
    def __init__(self,image_dir,heart_dir,llung_dir, rlung_dir,lcdir,rcdir,transforms = None):
        self.image_dir = image_dir
        self.heart_dir = heart_dir
        self.llung_dir = llung_dir
        self.rlung_dir = rlung_dir
        self.lcdir = lcdir
        self.rcdir = rcdir 
        self.transform = transforms
        
        self.image_filenames = sorted(os.listdir(self.image_dir))
        self.mask_filenames = sorted(os.listdir(self.heart_dir))

    
    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir,self.image_filenames[idx])
        image = Image.open(image_path).convert("L")
        
        heart_mask_path = os.path.join(self.heart_dir,self.mask_filenames[idx])
        heart_mask= cv2.imread(heart_mask_path, cv2.IMREAD_GRAYSCALE)
        #heart_mask = Image.open(heart_mask_path ).convert('L')
        
        left_lung_mask_path = os.path.join(self.llung_dir,self.mask_filenames[idx])
        right_lung_mask= cv2.imread(left_lung_mask_path, cv2.IMREAD_GRAYSCALE)
        #right_lung_mask= Image.open(left_lung_mask_path ).convert('L')
        
        right_lung_mask_path = os.path.join(self.rlung_dir,self.mask_filenames[idx])
        left_lung_mask = cv2.imread(right_lung_mask_path, cv2.IMREAD_GRAYSCALE)
        #left_lung_mask=Image.open(right_lung_mask_path ).convert('L')
        
        left_cl_mask_path = os.path.join(self.lcdir,self.mask_filenames[idx])
        left_clavicle_mask= cv2.imread(left_cl_mask_path, cv2.IMREAD_GRAYSCALE)
        #left_clavicle_mask= Image.open(left_cl_mask_path ).convert('L')
        
        right_cl_mask_path = os.path.join(self.rcdir,self.mask_filenames[idx])
        right_clavicle_mask= cv2.imread(right_cl_mask_path, cv2.IMREAD_GRAYSCALE)
        #right_clavicle_mask= Image.open(right_cl_mask_path ).convert('L')
        
        mask = multiclass_masks(heart_mask,right_lung_mask,left_lung_mask,right_clavicle_mask,left_clavicle_mask)
        
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, mask
    

    

def getdataset(base_path,tr_label,val_label,ts_label):
    train_dataset = segmentation_dataset(root_dir=base_path, data_file=tr_label, transform=transform)
    test_dataset =segmentation_dataset(root_dir=base_path, data_file=ts_label, transform=transform)
    val_dataset = segmentation_dataset(root_dir=base_path, data_file=val_label, transform=transform)
    return  train_dataset,test_dataset,val_dataset

def getdataloader(train_dataset,test_dataset,val_dataset,batch_size,num_gpus):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=num_gpus)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=num_gpus)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,num_workers=num_gpus)
    return train_loader,test_loader,val_loader


