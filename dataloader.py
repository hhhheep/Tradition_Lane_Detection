import albumentations as A
import cv2
import numpy as np
import torch, os, pdb
import random as rn
import os
# from torchvision import transforms
import pandas as pd
from tqdm import tqdm
from skimage.measure import label,regionprops
import pickle
import logging
# from multiprocessing import Pool
# from torch.utils.data import Dataset
# import glob
# from functools import partial
# from pathlib import Path
# from os.path import splitext,isfile,join
# from PIL import Image
# from os import listdir
import random



mean = (0.485, 0.456, 0.406)  # ImageNet 平均值
std = (0.229, 0.224, 0.225)   # ImageNet 標準差

train_aug = A.Compose([
    A.Resize(height=128, width=128), 
    # A.Normalize(mean=mean, std=std),
    A.RandomRotate90(p=0.5),  # 旋轉 +/- 10 度
    A.Affine(scale=(0.85, 1.15), shear=0.15, translate_percent=0.1), # 縮放比例 0.85-1.15, 剪切比例 0.15, 平移比例 0.1
    A.HorizontalFlip(p=0.5),  # 水平翻轉 (若需要的話請取消註解)
    A.VerticalFlip(p=0.5),    # 垂直翻轉 (若需要的話請取消註解)
], p=1.0)
test_aug = A.Compose([
    A.Resize(height=128, width=128), 
    # A.Normalize(mean=mean, std=std),
    # A.RandomRotate90(10),  # 旋轉 +/- 10 度
    # A.Affine(scale=(0.85, 1.15), shear=0.15, translate_percent=0.1), # 縮放比例 0.85-1.15, 剪切比例 0.15, 平移比例 0.1
    # A.HorizontalFlip(p=0.5),  # 水平翻轉 (若需要的話請取消註解)
    # A.VerticalFlip(p=0.5),    # 垂直翻轉 (若需要的話請取消註解)
], p=1.0)

# onsite_data_transforms =  \
#     A.Compose([
#             A.HorizontalFlip(p=0.5),
#             A.VerticalFlip(p=0.5),
#             A.Transpose(p=0.5),
#             A.RandomResizedCrop(height=512, width=512, scale=(0.75, 1.0), p=0.5),

#             ], p=1.0)
import pandas as pd                                     

class dataset_IMG(torch.utils.data.Dataset):

    def __init__(self,
                csv_path,
                dir_path="F:\Project-TrafficSign-Classifier", 
                train_model = "test",
                 
 
                   ):
       

        self.dir_path = dir_path
        self.csv_path = csv_path
        self.get_file()

        self.train_m = train_model

    

    def get_file(self):
        
        Data = pd.read_csv(os.path.join(self.dir_path,self.csv_path))
        self.img_list =  Data[["Path"]].values.T[0]
        self.label_list = Data[["ClassId"]].values.T[0]

    def __len__(self):
        return len(self.img_list)
    
    

    def __getitem__(self,index):
        img = cv2.imread(self.img_list[index])
        label = self.label_list[index] #np.load(self.label_path.joinpath(self.img_list[index]))
        # pre_lab = self.pre_pred["pre_label"].values[index]
        # H,W,C = img.shape

       

      
       
        if self.train_m == "train":

        #     if  not self.aug_select is None:
        #         for indx,tr in enumerate(self.aug_select):
                
        #             img[:,:,indx] = self.aug_band(img,select = tr,index=indx)

            transformed_data = train_aug(image=img)

            img = transformed_data["image"]
        #     label = transformed_data["mask"]

        else:
            
            transformed_data = test_aug(image=img)
            img = transformed_data["image"]
        #   
        img = torch.from_numpy(img.transpose((2, 0, 1)))
        label = torch.tensor(label)

        # img = torch.tensor(img)
        # label = torch.tensor(label)

        return img,label

    # def scale_image(self, img):
    
    #     # scaled_img = cv2.resize(img, None, fx=self.img_scale, fy=self.img_scale)
    #     # 将图像转换为 PyTorch Tensor 格式
    #     # scaled_img = scaled_img.reshape((scaled_img.shape[0],scaled_img.shape[1],scaled_img.shape[2])) #if len(scaled_img.shape)<3 else scaled_img
    #     # normalized_image = scaled_img / 255.0

    #     tensor_img = 
    #     return tensor_img