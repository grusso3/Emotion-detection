import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module, BatchNorm2d
import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import torch
import torchvision.transforms as transforms


class EmotionData(Dataset):
    def _init_(self, csv_file, root_dir):
        self.annotations = pd.read_csv(csv_file)  # pd object
        self.root_dir = root_dir

    def _len_(self):
        return len(self.annotations)

    def _getitem_(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index][0])
        image = Image.open(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index][1])).long()

        transform_train = transforms.Compose([
            transforms.ToTensor()
        ])
        image = transform_train(image)

        #return image, y_label     # for our network, hash it while training VGG

        return image.repeat(3,1,1), y_label # for VGG, hash it while training our own Nets

class EmotionDataTest(Dataset):
    def _init_(self, csv_file, root_dir):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def _len_(self):
        return len(self.annotations)

    def _getitem_(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index][0])
        image = Image.open(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index][1])).long()


        transform = transforms.ToTensor()
        image = transform(image)
        #return image, y_label    # for our network, hash it while training VGG

        return image.repeat(3, 1, 1), y_label # for VGG, hash it while training our own Nets



class Network(nn.Module):
    def _init_(self):
        super()._init_()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(4, 4))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(3, 3))
        self.fc1 = nn.Linear(in_features=12*9*9, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=7)

    def forward(self, t):
        # input layer
        t = t

        # hidden conv layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=4, stride=2)

        # 2nd hidden conv layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=3, stride=2)

        # 3rd hidden layer
        t = t.reshape(-1, 12*9*9)
        t = self.fc1(t)
        t = F.relu(t)

        # 4th hidden layer
        t = self.fc2(t)
        t = F.relu(t)

        # Output layer
        t = self.out(t)

        t = F.softmax(t, dim = 1)
        # already included in cross entrophy loss
        return t


class Network2(nn.Module):
    def _init_(self):
        super()._init_()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.fc1 = nn.Linear(in_features= 128 * 42 * 42, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.out = nn.Linear(in_features=256, out_features=7)

    def forward(self, t):
        # input layer
        t = t

        # hidden conv layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=3, stride=1)

        # 2nd hidden conv layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=3, stride=1)

        # 3nd hidden conv layer
        t = self.conv3(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=3, stride=1)

        # 3rd hidden layer
        t = t.reshape(-1, 128 * 42 * 42)
        t = self.fc1(t)
        t = F.relu(t)

        # 4th hidden layer
        t = self.fc2(t)
        t = F.relu(t)

        # Output layer
        t = self.out(t)

        t = F.softmax(t, dim = 1 )
        # already included in cross entrophy loss
        return t
