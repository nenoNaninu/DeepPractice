# -*- using:utf-8 -*-
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import cv2
import torch


class PascalVOC(Dataset):
    def __init__(self, img_list_txt_path, mask_list_path, img_size):
        with open(img_list_txt_path) as f:
            # 行ごとにすべて読み込んでリストデータにする
            self.img_file_path_list = f.read().splitlines()

        with open(mask_list_path) as f:
            self.mask_path_list = f.read().splitlines()

        self.img_size = img_size

        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __len__(self):
        return len(self.mask_path_list)

    def __getitem__(self, idx):
        img_path = self.img_file_path_list[idx]
        segment_path = self.mask_path_list[idx]

        # print(img_path)
        img = cv2.imread(img_path)
        # print(img)
        # print(type(img))
        # print(img.shape)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)

        segment_img = np.loadtxt(segment_path)
        segment_img = cv2.resize(
            segment_img, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        class_num = 20
        segment_mask = np.zeros(
            (class_num, self.img_size, self.img_size), dtype=np.float32)
        for i in range(class_num):
            idxes = segment_img == i
            segment_mask[i, idxes] = 1

        return img, torch.from_numpy(segment_mask)


def test():
    img_list_path = "/export/space0/shimoda-k/wseg/data/img_list.txt"
    mask_list_path = "/export/space0/shimoda-k/wseg/data/mask_list.txt"
    pascalVOC = PascalVOC(img_list_path, mask_list_path, img_size=224)
    img, segment_img = pascalVOC[21]
    # print(img)
    # print(segment_img)
    # print(img.shape)
    # print(segment_img.shape)


if __name__ == "__main__":
    test()
