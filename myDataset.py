from torch.utils.data import Dataset,DataLoader,random_split
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

class myDataset(Dataset):   # 自定义的数据读取类，继承torch.utils.data.Dataset类
    def __init__(self,transform=None):
        images_path = "./data/images"

        self.file_path_list = []
        for root, dirs, file in os.walk(images_path):
            self.file_path_list.extend(file)
        for i in range(len(self.file_path_list)):
            self.file_path_list[i] = images_path + "/" + self.file_path_list[i][8] + "/" + self.file_path_list[i]
        self.df = pd.read_excel("./data/labels.xlsx")
        self.transform = transform
    def __getitem__(self,index):    # 输入index索引，获得对应图像和标签
        path = self.file_path_list[index]
        patient_ID = int(path[16:23])

        # 读取图像
        data_arr = np.load(path).astype(np.float32)
        # 手动ToTensor   2.归一化
        data_arr = data_arr.transpose((2, 0, 1))  # 1.HWC -> CHW
        lymph1, lymph2, tumor1, tumor2 = data_arr[0, :, :], data_arr[1, :, :], data_arr[2, :, :], data_arr[3, :, :]
        if lymph1.max() != lymph1.min():
            lymph1 = (lymph1 - lymph1.min()) / float(lymph1.max() - lymph1.min())
        if lymph2.max() != lymph2.min():
            lymph2 = (lymph2 - lymph2.min()) / float(lymph2.max() - lymph2.min())
        if tumor1.max() != tumor1.min():
            tumor1 = (tumor1 - tumor1.min()) / float(tumor1.max() - tumor1.min())
        if tumor2.max() != tumor2.min():
            tumor2 = (tumor2 - tumor2.min()) / float(tumor2.max() - tumor2.min())
        data_arr = np.vstack([lymph1, lymph2, tumor1, tumor2]).reshape(4, 194, 194)
        data_arr = torch.from_numpy(data_arr)
        # data_arr = Image.fromarray(data_arr)  # 这里ndarray_image为原来的numpy数组类型的输入
        # if self.transform:
        #     data_arr = self.transform(data_arr)

        # print(patient_ID)
        label = self.df.loc[self.df["病人ID"]==patient_ID].iloc[:,3].iloc[0]
        # 合并3，4为一类
        if label > 3:
            label -= 1
        label = torch.from_numpy(np.array(label)).long()
        return data_arr,label
    def __len__(self):
        return len(self.file_path_list)