from data.base_dataset import BaseDataset, get_transform
from data.medicine_image_folder import make_dataset
import numpy as np
import torch
from PIL import Image
import pydicom
import matplotlib.pyplot as plt

class MedicineDataset(BaseDataset):
    def __init__(self,opt):

        BaseDataset.__init__(self,opt)
        self.A_paths = sorted(make_dataset(opt.dataroot_A,opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(opt.dataroot_B, opt.max_dataset_size))
        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.transform = get_transform(opt, grayscale=(input_nc == 1))

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        A_img = pydicom.dcmread(A_path).pixel_array
        #print('A_path:',A_path,'   ','A_img:',type(A_img))
        #A_img = np.moveaxis(A_img, 0, 1)
        A_img =(A_img - A_img.min(axis=0))/( A_img.max(axis=0) - A_img.min(axis=0) )
        A_img = Image.fromarray(A_img,mode='RGB')
        A = self.transform(A_img)
        #print(A)
        plt.figure(figsize=(10,10))
        plt.imshow(A_img,cmap=plt.cm.bone)
        plt.show()

        B_path = self.B_paths[index]
        B_img = pydicom.dcmread(B_path).pixel_array
        B_img = (B_img - B_img.min(axis=0))/ (B_img.max(axis=0) - B_img.min(axis=0))
        B_img = Image.fromarray(B_img, mode='RGB')

        B = self.transform(B_img)

        return {'A':A,'A_paths':A_path,'B':B,'B_paths':B_path}

    def __len__(self):
        return len(self.A_paths)