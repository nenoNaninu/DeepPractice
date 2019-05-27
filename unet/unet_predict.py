# -*- using:utf-8 -*-
from unet import UNet
from pascal_dataset import PascalVOC
import os
from torch.utils.data import DataLoader
import torch
from torch import nn
from datetime import datetime
from PIL import Image
import numpy as np


def get_palette(class_num):
    """ Returns the color palette for visualizing the segmentation mask.
    Args:
        class_num: Number of classes
    Returns:
        The color palette
    """
    n = class_num
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def mask2png(mask, class_num, save_dir=os.getcwd(), save_name=datetime.now().strftime("%Y%m%d%H:%M:%S") + ".png"):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        mask: mask as numpy
        class_num: class num

    Returns:
        The color map
    """
    palette = get_palette(class_num)
    mask = Image.fromarray(mask.astype(np.uint8))
    mask.putpalette(palette)
    mask.save(os.path.join(save_dir, save_name))
    return mask


if __name__ == "__main__":
    # 以下イメージ
    img_list_path = "/export/space0/shimoda-k/wseg/data/img_list.txt"
    mask_list_path = "/export/space0/shimoda-k/wseg/data/mask_list.txt"
    checkpoint_path = os.path.join(os.getcwd(), "unet_train2/CP250.prm")
    output_dir = os.path.join(os.getcwd(), "unet_prediction_output")

    print(checkpoint_path)

    pascalVOC = PascalVOC(img_list_path, mask_list_path, img_size=224)

    batch_size = 10
    trainloader = DataLoader(
        pascalVOC, batch_size=batch_size, shuffle=True, num_workers=6)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    unet = UNet(3, 20)
    unet = nn.DataParallel(unet)
    unet.load_state_dict(torch.load(checkpoint_path))
    unet.to(device)

    idx = 2
    img_data = pascalVOC[idx][0]

    output = unet(img_data.unsqueeze(0)).squeeze(0)

    output = output.detach().cpu().numpy().transpose(1, 2, 0).argmax(2)

    mask2png(output, 20, save_name="mask{0}.png".format(idx))

    img_data = img_data.detach().cpu().numpy()
    img_data = (img_data - np.min(img_data)) / \
        (np.max(img_data) - np.min(img_data))
    img = Image.fromarray((img_data * 255).astype(np.uint8).transpose(1, 2, 0))
    img.save("./original{0}.png".format(idx))
