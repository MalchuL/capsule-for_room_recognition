import pandas as pd
import PIL.Image
import numpy as np
import os
import torchvision.transforms as tr
from torchvision.transforms import *
import torch.utils.data
from torchvision.datasets import ImageFolder



class HackatonDataset(torch.utils.data.Dataset):
    """Dataset wrapping images and target labels for Kaggle - Planet Amazon from Space competition.

    Arguments:
        A CSV file path
        Path to image folder
        Extension of images
        PIL transforms
    """

    def __init__(self, img_path, img_ext):
        # ToDo check for names_path end with .txt else load all files from names_path
        # "Some images referenced in the CSV file were not found"
        self.img_path = img_path
        self.img_ext = img_ext
        self.transform = tr.Compose(
            [tr.Resize((299, 299)), tr.RandomAffine(25), tr.ColorJitter(0.3, 0.5, 0.1), tr.RandomHorizontalFlip(),
             tr.ToTensor()])

    def get_size(self):
        return len(self.X_train)

    """def __getitem__(self, index):
        img = PIL.Image.open(self.img_path + self.X_train[index] + self.img_ext)
        img = img.convert('RGB')
        cln = self.img_path + folder_name
        if self.transform is not None:
            img = self.transform(img)
        return img, cln"""

    def get_classes(self, classname, count_of_choices,X,Y):
        root='./datasets/'+classname+'/'
        number_of_files = len([item for item in os.listdir(root+'/'+classname+'/') if os.path.isfile(os.path.join(root+'/'+classname+'/', item))])
        imfolder = ImageFolder(root=root, transform=self.transform)
        array = np.random.choice(number_of_files, count_of_choices)
        print(number_of_files[array[0]])
        for element in array:
            x=imfolder.__getitem__(element)
            X.append(x[0])
            y=classname
            Y.append(int(y))

        return X, Y

    def get_train_batch(self, batch_size):
        X, Y=[],[]
        self.get_classes('0', batch_size//5, X, Y)
        self.get_classes('1', batch_size//5, X, Y)
        self.get_classes('2', batch_size//5, X, Y)
        self.get_classes('3', batch_size//5, X, Y)
        self.get_classes('4', batch_size//5, X, Y)
        X = torch.stack(X, 0).type(torch.FloatTensor)
        return X, Y