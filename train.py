import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, Conv2d, BatchNorm2d
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,random_split
from torchvision import models,transforms
from sklearn.metrics import recall_score,f1_score
import time
import cv2
import matplotlib.pyplot as plt
import joblib
from myDataset import myDataset

def learning_curve(record_train):   # 画出不用epoch损失函数变化的图，并保存
    plt.style.use("ggplot")

    plt.plot(range(1, len(record_train) + 1), record_train, label="train loss")


    plt.legend(loc=4)
    plt.title("train loss")

    plt.xlabel("epoch")
    plt.ylabel("train loss")

    plt.savefig("./result/train_loss.png")
    plt.show()

#定义评估的函数
#准确率作为评估标准
def accuracy(predictions,labels):
    pred=torch.max(predictions.data,1)[1]
    rights=pred.eq(labels.data.view_as(pred)).sum()
    return rights,len(pred)


batchsize = 40   # 每个批次训练的样本数，可以适当调大
num_epoch = 300

# 加载训练数据集，transforms.Compose 是数据增强
my_dataset = myDataset(transform=transforms.Compose([
                          transforms.RandomHorizontalFlip(),
                          transforms.RandomVerticalFlip(),
                          transforms.RandomRotation(30),
                          transforms.CenterCrop(80),
                          # transforms.ToTensor(),
                      ]))
# 随机划分数据集
train_dataset, test_dataset = random_split(
    dataset=my_dataset,
    lengths=[int(0.8*my_dataset.__len__()), my_dataset.__len__()-int(0.8*my_dataset.__len__())],    # 80%是训练集
    generator=torch.manual_seed(0)
)

# 加载数据
train_dataLoader = DataLoader(dataset=train_dataset,batch_size=batchsize,shuffle=True)
test_dataLoader = DataLoader(dataset=test_dataset,batch_size=my_dataset.__len__()-int(0.8*my_dataset.__len__()),shuffle=False)

#指定设备device为cuda
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

net = models.alexnet(pretrained=True)  # alexnet
net.features[0] = Sequential(Conv2d(4, 64, kernel_size=11, stride=4, padding=2))  # alexnet
net.classifier[6] = Sequential(Linear(4096, 4))
torch.save(net,"./weight/alexnet.pkl")   # 保存网络模型
net = net.to(device)

#损失函数
# criterion=nn.MultiMarginLoss()
criterion=nn.CrossEntropyLoss()
#优化器
optimizer=optim.Adam(net.parameters(),lr=0.0001)

# 训练
print("training...")
train_loss_list = [] # 用于保存每个epoch的损失值
for epoch in range(num_epoch):
    train_epoch_loss = 0
    # train_rights用于将当前epoch的准确率保存下来
    train_rights = []
    for idx,(data,target) in enumerate(train_dataLoader):
        start_time = time.time()
        data = data.to(device)
        target = target.to(device)

        net.train()
        output = net(data)
        # print(output.shape)
        # print(target.shape)
        # print(target)
        loss = criterion(output, target)    # 计算损失值
        train_epoch_loss += loss.cpu().detach().numpy()

        optimizer.zero_grad()   # 优化器梯度清零
        loss.backward()     # 反向传播
        optimizer.step()    # 更新网络参数
        right = accuracy(output,target)     # 计算当前训练准确率
        train_rights.append(right)

        # 测试
        net.eval()
        val_rights=[]   # 用于保存验证集的准确率

        for (data,target) in test_dataLoader:
            data = data.to(device)
            target = target.to(device)

            # net.cuda()
            output = net(data)
            pred = torch.max(output.data, 1)[1]     # 由output得到预测结果
            uar = recall_score(target.cpu().detach().numpy(), pred.cpu().detach().numpy(), average='macro')     # UAR

            uf1 = f1_score(target.cpu().detach().numpy(), pred.cpu().detach().numpy(), average='macro')     # UF1

            right = accuracy(output, target)
            val_rights.append(right)

        # 准确率计算
        train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
        val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))

        print('当前epoch:{} [{}/{} ({:.0f}%)]\t损失：{:.6f}\t训练集准确率：{:.2f}%\t测试集正确率：{:.2f}% uar：{:.3f} uf1：{:.3f}\t用时{:.2f}s'.format(
            epoch+1, idx * batchsize, len(train_dataLoader.dataset),
                   100. * idx / len(train_dataLoader),
            loss.data,
                   100. * train_r[0].cpu().numpy() / train_r[1],
                   100. * val_r[0].cpu().numpy() / val_r[1],uar,uf1,time.time()-start_time
        ))
    train_loss_list.append(train_epoch_loss)
    torch.save(net,"./weight/alexnet.pkl")   # 保存网络模型

learning_curve(train_loss_list)
torch.save(net, "./weight/alexnet.pkl")  # 保存网络模型
print("Sucessfully saved!")