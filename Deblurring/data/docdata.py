import os

import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, RandomAffine, RandomHorizontalFlip, RandomCrop
from PIL import Image
import numpy as np

def ImageTransform(loadSize):
    return {"train": Compose([
        RandomCrop(loadSize, pad_if_needed=True, padding_mode='constant', fill=255),
        RandomAffine(10, fill=255),
        RandomHorizontalFlip(p=0.2),
        ToTensor(),
    ]), "test": Compose([
        ToTensor(),
    ]), "train_gt": Compose([
        RandomCrop(loadSize, pad_if_needed=True, padding_mode='constant', fill=255),
        RandomAffine(10, fill=255),
        RandomHorizontalFlip(p=0.2),
        ToTensor(),
    ])}


class DocData(Dataset):
    def __init__(self, path_img, path_gt, loadSize, mode=1):
        super().__init__()
        self.path_gt = path_gt
        self.path_img = path_img
        self.data_gt = sorted(os.listdir(path_gt))
        self.data_img = sorted(os.listdir(path_img))
        self.mode = mode
        if mode == 1:
            self.ImgTrans = (ImageTransform(loadSize)["train"], ImageTransform(loadSize)["train_gt"])
        else:
            self.ImgTrans = ImageTransform(loadSize)["test"]

    def __len__(self):
        return len(self.data_gt)

    def __getitem__(self, idx):

        # gt = Image.open(os.path.join(self.path_gt, self.data_gt[idx]))
        gt = self.enhance_text_clarity(os.path.join(self.path_gt, self.data_gt[idx]))
        img = Image.open(os.path.join(self.path_img, self.data_img[idx]))
        img = img.convert('RGB')
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        if self.mode == 1:
            seed = torch.random.seed()
            torch.random.manual_seed(seed)
            img = self.ImgTrans[0](img)
            torch.random.manual_seed(seed)
            gt = self.ImgTrans[1](gt)
        else:
            img= self.ImgTrans(img)
            gt = self.ImgTrans(gt)
        name = self.data_img[idx]
        #print(f"Processing: {name}\n")
        return img, gt, name

    def enhance_text_clarity(self, image_path):
        # 读取图像
        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(f"图像文件 {image_path} 未找到")

        # 双边滤波
        bilateral = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

        # 自适应直方图均衡化
        lab = cv2.cvtColor(bilateral, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # 应用锐化滤波器
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        return sharpened