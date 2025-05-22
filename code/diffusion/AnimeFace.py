import torch
import torchvision
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from matplotlib import pyplot as plt
import random
import numpy as np
import os
import glob
#TODO: failed

def get_img_from_path(path:str) -> torch.Tensor:
    size =  os.path.getsize(path)
    byte_datas = torch.from_file(path,size=size,dtype=torch.uint8)
    img = torchvision.io.decode_jpeg(byte_datas,torchvision.io.ImageReadMode.RGB)
    img = (img.squeeze(0)/255)*2 -1
    return img



class AnimeFace(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        filepaths = glob.glob("../data/AnimeFace/images/*.jpg")
        self.data = []
        for path in filepaths[:2000]:
            self.data.append(get_img_from_path(path=path))
        self.n = len(self.data)
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return self.n
    