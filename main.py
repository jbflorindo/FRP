from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torchvision import models, transforms
from datasets import list_images,GeneralDataset,read_kth

#%% Training

# Training
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):

    best_acc = 0.0
    train_acc_list, test_acc_list = [], []

    for epoch in range(num_epochs):

        all_preds, all_labels = [],[]

        for phase in ['training', 'testing']:
            if phase == 'training':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                def closure():
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    return loss

                with torch.set_grad_enabled(phase == 'training'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)
                    
                    if phase == 'testing':
                        all_preds.append(preds.cpu().detach().numpy())
                        all_labels.append(labels.data.cpu().detach().numpy())

                    if phase == 'training':
                        loss.backward()
                        optimizer.step(closure)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            if phase == 'training':
                train_acc_list.append(epoch_acc)
            if phase == 'testing':
                test_acc_list.append(epoch_acc)

            if phase == 'testing' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                y_test = np.concatenate(all_labels)
                y_pred = np.concatenate(all_preds)
                
        print("Epoch: ", epoch)

    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best epoch: {}'.format(best_epoch))

    return best_acc, y_test, y_pred, train_acc_list, test_acc_list

# Parameters to be updated
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

#%% Fractal module
    
import math
# Fractal dimension module (based on Xu, Y., Li, F., Chen, Z., Liang, J., & Quan, Y. (2021). Encoding spatial distribution of convolutional features for texture representation. Advances in Neural Information Processing Systems, 34, 22732-22744.)
class GDCB(nn.Module):
    def __init__(self,mfs_dim=25,nlv_bcd=6):
        super(GDCB,self).__init__()
        self.mfs_dim=mfs_dim
        self.nlv_bcd=nlv_bcd
        self.pool=nn.ModuleList()
        
        for i in range(self.nlv_bcd-1):
            self.pool.add_module(str(i),nn.MaxPool2d(kernel_size=i+2,stride=(i+2)//2))
        self.ReLU = nn.ReLU()
    def forward(self,input):
        tmp=[]
        for i in range(self.nlv_bcd-1):
            output_item=self.pool[i](input)
            tmp.append(torch.sum(torch.sum(output_item,dim=2,keepdim=True),dim=3,keepdim=True))
        output=torch.cat(tuple(tmp),2)#why 0 appear
        output=torch.log2(self.ReLU(output)+1)
        X=[-math.log(i+2,2) for i in range(self.nlv_bcd-1)]
        X = torch.tensor(X).to(output.device)
        X=X.view([1,1,X.shape[0],1])
        meanX = torch.mean(X,2,True)
        meanY = torch.mean(output,2,True)
        Fracdim = torch.div(torch.sum((output-meanY)*(X-meanX),2,True),torch.sum((X-meanX)**2,2,True))
        return Fracdim 

#%% Dataset processing

Experiment_acc_history = []
dataset = 'FMD'
batch_size=32
dropout_ratio=0.6
input_size=224
learning_ratio=0.0001
num_epochs = 100
acc_exp_list = []

experiment_idx = 0
print("RUN: "+str(experiment_idx))

data_transforms = {
    'training': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    'testing': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
os.getcwd()
should_data_augmentation=True
experiment_idx_str=str(experiment_idx)
if dataset == 'FMD':
    train_imgs,val_imgs,train_classes,val_classes,classes = list_images('FMD')
    image_datasets = {'training' : GeneralDataset('fmd'+os.sep+'image',train_imgs, train_classes, transform=data_transforms['training']), 'testing': GeneralDataset('fmd'+os.sep+'image',val_imgs, val_classes, transform=data_transforms['testing'])}
if dataset == 'KTH':
    train_imgs,val_imgs,train_classes,val_classes,classes = read_kth(experiment_idx)
    image_datasets = {'training' : GeneralDataset('kth',train_imgs, train_classes, transform=data_transforms['training']), 'testing': GeneralDataset('kth',val_imgs, val_classes, transform=data_transforms['testing'])}
if dataset == '1200tex':
    train_imgs,val_imgs,train_classes,val_classes,classes = list_images('1200tex')
    image_datasets = {'training' : GeneralDataset('1200Tex',train_imgs, train_classes, transform=data_transforms['training']), 'testing': GeneralDataset('1200Tex',val_imgs, val_classes, transform=data_transforms['testing'])}
if dataset == 'uiuc':
    train_imgs,val_imgs,train_classes,val_classes,classes = list_images('uiuc')
    image_datasets = {'training' : GeneralDataset('uiuc',train_imgs, train_classes, transform=data_transforms['training']), 'testing': GeneralDataset('uiuc',val_imgs, val_classes, transform=data_transforms['testing'])}
if dataset == 'umd':
    train_imgs,val_imgs,train_classes,val_classes,classes = list_images('umd')
    image_datasets = {'training' : GeneralDataset('umd',train_imgs, train_classes, transform=data_transforms['training']), 'testing': GeneralDataset('umd',val_imgs, val_classes, transform=data_transforms['testing'])}        
class_names = classes
dataset_sizes = {x: len(image_datasets[x]) for x in ['training', 'testing']}
print(dataset_sizes)
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in ['training', 'testing']}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_classes=len(class_names)
feature_extract=True

class myModel(nn.Module):
    def __init__(self):
        super(myModel,self).__init__()
        model_dense=models.densenet161(pretrained=True)
        self.features=nn.Sequential(*list(model_dense.features.children())[:-1])            
        # self.conv1= nn.Sequential(nn.Conv2d(in_channels=2208,
        #                                 out_channels=1104,
        #                                   kernel_size=1,
        #                                 stride=1,
        #                                 padding=0),
        #                     nn.Dropout2d(p=0.5),
        #                       nn.BatchNorm2d(1104))
                    
        # self.relu1 = nn.ReLU(inplace=True)
        # self.relu2 = nn.ReLU(inplace = True)
        # self.norm1 = nn.BatchNorm2d(4416);
        # self.relu3 = nn.ReLU(inplace=True);
        # self.classifier=nn.Linear((4416),num_classes)

    def forward(self,x):

        out = self.features(x)
        identity=out
        identity = self.sigmoid(identity)                
        out = self.conv1(out)
        out = self.relu1(out)
        out = out-identity # Residual module          
        out1 = nn.functional.adaptive_avg_pool2d(out,(1,1)).view(out.size(0), -1) 
        box_count = nn.Sequential(GDCB())
        out2 = box_count(out).view(out.size(0), -1) # Fractal pooling
        out3 = out1*out2
        x=self.classifier(out3)
        return x

net=myModel()
feature_extract=True
set_parameter_requires_grad(net, feature_extract)
dense_feature_dim = 2208
net.conv1= nn.Sequential(nn.Conv2d(in_channels=dense_feature_dim,
                                        out_channels=dense_feature_dim,
                                          kernel_size=1,
                                        stride=1,
                                        padding=0),
                            nn.Dropout2d(p=dropout_ratio),
                              nn.BatchNorm2d(dense_feature_dim))
net.sigmoid=nn.Sigmoid()
net.relu1 = nn.Sigmoid()
net.relu2 = nn.ReLU(inplace=True);
net.norm1 = nn.BatchNorm2d(dense_feature_dim);
net.relu3 = nn.ReLU(inplace=True);
net.classifier=nn.Linear((dense_feature_dim),num_classes)
criterion = nn.CrossEntropyLoss()
model_ft = net.to(device)
params_to_update = model_ft.parameters()

if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)

classifier_params = list(map(id, model_ft.classifier.parameters()))
base_params = filter(lambda p: id(p) not in classifier_params,
                      params_to_update)

optimizer_ft = optim.Adam(params_to_update,lr=0.0001)
best_acc,y_test,y_pred,train_acc_list,test_acc_list = train_model(model_ft, dataloaders, criterion, optimizer_ft, num_epochs=num_epochs)
acc_exp_list.append(best_acc)
np.save('acc_exp_list.npy', torch.FloatTensor(acc_exp_list))       