import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transforms_ = transforms_  # Сохраняем transforms_ как есть, без Compose

        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        if mode == "train":
            self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w // 2, h))  # Используем целочисленное деление
        img_B = img.crop((w // 2, 0, w, h))

        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

        # Применяем transforms_, если они есть
        if self.transforms_ is not None:
            img_A = self.transforms_(img_A)
            img_B = self.transforms_(img_B)

        return {"A": img_A, "B": img_B}

    def __len__(self):
        return len(self.files)