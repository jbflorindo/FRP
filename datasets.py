import cv2
from torch.utils.data import Dataset
import glob
from PIL import Image
import os

# FMD, KTH
from sklearn.model_selection import train_test_split

class GeneralDataset(Dataset):
    def __init__(self, img_dir, img_list, class_list, transform=False):
        self.img_dir = img_dir
        self.img_list = img_list
        self.class_list = class_list
        self.transform = transform
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        image_filepath = os.path.join(self.img_dir,self.img_list[idx])
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        
        label = self.class_list[idx]
        
        if self.transform is not None:
            image = self.transform(image)

        return image, label
    
class GeneralDataset_per_case(Dataset):
    def __init__(self, img_dir, img_list, class_list, case_list, transform=False):
        self.img_dir = img_dir
        self.img_list = img_list
        self.class_list = class_list
        self.case_list = case_list
        self.transform = transform
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        image_filepath = os.path.join(self.img_dir,self.img_list[idx])
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        
        label = self.class_list[idx]
        
        case = self.case_list[idx]
        
        if self.transform is not None:
            image = self.transform(image)

        return image, label, case    

def list_images(database):
    
    if database == 'FMD':
        dir_database = 'fmd'+os.sep+'image'
        extension = os.sep+'*.jpg'

    if database == 'KTH':
        dir_database = 'kth'
        extension = os.sep+'*.png'

    if database == '1200tex':
        dir_database = '1200Tex'
        extension = os.sep+'*.png'
        
    if database == 'uiuc':
        dir_database = 'uiuc'
        extension = os.sep+'*.jpg'

    if database == 'umd':
        dir_database = 'umd'
        extension = os.sep+'*.png'        

    classes = [] #to store class values
    images = [] #to store image names
    for data_path in glob.glob(dir_database+os.sep+'*'+os.sep):
        classes.append(data_path.split(os.sep)[-2]) 
        if database == 'KTH':
            sample_list = ['sample_a','sample_b','sample_c','sample_d']
            for sample in sample_list:
                for data_path2 in glob.glob(data_path + sample + extension):                    
                    images.append(os.path.join(data_path2.split(os.sep)[-3],data_path2.split(os.sep)[-2],data_path2.split(os.sep)[-1]))             
        else:
            for data_path2 in glob.glob(data_path + extension):
                images.append(os.path.join(data_path2.split(os.sep)[-2],data_path2.split(os.sep)[-1]))             

    idx_to_class = {i:j for i, j in enumerate(classes)}
    class_to_idx = {value:key for key,value in idx_to_class.items()}            

    image_class = [] #class of each image
    for data_path in glob.glob(dir_database + os.sep+'*'+os.sep):
        if database == 'KTH':
            sample_list = ['sample_a','sample_b','sample_c','sample_d']
            for sample in sample_list:
                for data_path2 in glob.glob(data_path + os.sep + sample + extension):
                    image_class.append(class_to_idx[data_path.split(os.sep)[-2]])
        else:
            for data_path2 in glob.glob(data_path + extension):
                image_class.append(class_to_idx[data_path.split(os.sep)[-2]])

    train_imgs, val_imgs, train_classes, val_classes = train_test_split(images, image_class, test_size=0.5)
    
    return train_imgs,val_imgs,train_classes,val_classes,classes

def read_kth(sample_idx):

    dir_database = 'kth'
    extension = os.sep+'*.png'
    sample_list = ['sample_a','sample_b','sample_c','sample_d']
    to_exclude = {sample_idx}
    sample_list2 = [element for i, element in enumerate(sample_list) if i not in to_exclude]

    train_imgs, val_imgs, train_classes, val_classes, classes = [],[],[],[],[]
    for data_path in glob.glob(dir_database + os.sep+'*'+os.sep):
        classes.append(data_path.split(os.sep)[-2])
        for data_path2 in glob.glob(data_path + sample_list[sample_idx] + extension):
            val_imgs.append(os.path.join(data_path2.split(os.sep)[-3],data_path2.split(os.sep)[-2],data_path2.split(os.sep)[-1]))
        for sample in sample_list2:
            for data_path2 in glob.glob(data_path + sample + extension):
                train_imgs.append(os.path.join(data_path2.split(os.sep)[-3],data_path2.split(os.sep)[-2],data_path2.split(os.sep)[-1]))            

    idx_to_class = {i:j for i, j in enumerate(classes)}
    class_to_idx = {value:key for key,value in idx_to_class.items()}

    for data_path in glob.glob(dir_database + os.sep+'*'+os.sep):
        for data_path2 in glob.glob(data_path + os.sep + sample_list[sample_idx] + extension):
            val_classes.append(class_to_idx[data_path.split(os.sep)[-2]])
        for sample in sample_list2:
            for data_path2 in glob.glob(data_path + os.sep + sample + extension):
                train_classes.append(class_to_idx[data_path.split(os.sep)[-2]])

    return train_imgs,val_imgs,train_classes,val_classes,classes

import random
def read_kth_random(): # two random samples for training

    dir_database = 'kth'
    extension = os.sep+'*.png'
    sample_list = ['sample_a','sample_b','sample_c','sample_d']
    random.shuffle(sample_list)
    sample_list_train = sample_list[:2]
    sample_list_val = sample_list[2:]

    train_imgs, val_imgs, train_classes, val_classes, classes = [],[],[],[],[]
    for data_path in glob.glob(dir_database + os.sep+'*'+os.sep):
        classes.append(data_path.split(os.sep)[-2])
        for sample in sample_list_train:
            for data_path2 in glob.glob(data_path + sample + extension):
                train_imgs.append(os.path.join(data_path2.split(os.sep)[-3],data_path2.split(os.sep)[-2],data_path2.split(os.sep)[-1]))
        for sample in sample_list_val:
            for data_path2 in glob.glob(data_path + sample + extension):
                val_imgs.append(os.path.join(data_path2.split(os.sep)[-3],data_path2.split(os.sep)[-2],data_path2.split(os.sep)[-1]))            

    idx_to_class = {i:j for i, j in enumerate(classes)}
    class_to_idx = {value:key for key,value in idx_to_class.items()}

    for data_path in glob.glob(dir_database + os.sep+'*'+os.sep):
        for sample in sample_list_train:
            for data_path2 in glob.glob(data_path + os.sep + sample + extension):
                train_classes.append(class_to_idx[data_path.split(os.sep)[-2]])
        for sample in sample_list_val:
            for data_path2 in glob.glob(data_path + os.sep + sample + extension):
                val_classes.append(class_to_idx[data_path.split(os.sep)[-2]])

    return train_imgs,val_imgs,train_classes,val_classes,classes