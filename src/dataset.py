from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import torch
import torchvision.transforms as transforms


class EmotionData(Dataset):
    def __init__(self, csv_file, root_dir):
        self.annotations = pd.read_csv(csv_file)  # pd object
        self.root_dir = root_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index][0])
        image = Image.open(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index][1])).long()

        # Image 2 tensor
        trans = transforms.ToTensor()
        image = trans(image)
        return image, y_label
