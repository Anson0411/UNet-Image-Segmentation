import os 
import cv2 # 不支援gif
import numpy as np
from PIL import Image

from torch.utils.data import Dataset


class CarDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace('.jpg', '_mask.gif'))
        image = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path))
        return self.transform(image), self.transform(mask)
        