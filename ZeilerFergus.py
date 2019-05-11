# -*- using:utf-8 -*-
import time
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
import torch.optim as optim
import cv2
import numpy as np
import json
from tqdm import tqdm


# # 適当に畳み込みしていく。
# class VGGNet(nn.Module):
#     def __init__(self, num_classes):
#         super(VGGNet, self).__init__()
#
#     def forward(self, input):
#         output = output.view(-1, 32 * 16 * 16)
#
#         output = self.fc(output)
#
#         return output


def forward(model, img):
    model.eval()
    with  torch.no_grad():
        output = model(img)
        print("output shape:", output.shape)

        idx = torch.max(output, 1)[1]
        probability = output[:, idx]
        probability = probability.cpu().numpy()
        idx = idx.cpu().numpy()
        print("probability:", probability)
        print("idx:", idx)
        output = output.cpu().numpy()
    return probability, idx, output


def convert_to_intpu_tensor(img):
    # img = img.transpose(2, 0, 1)
    # img = img.astype(np.float32)
    # print("oorigin", img)
    # # img = img / 255.0
    # img = torch.from_numpy(img)

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # img_pil = Image.open("cat2.jpg")

    # img_tensor = preprocess(img)

    print("origin", img)
    print("norm", preprocess(img))
    return preprocess(img)


if __name__ == "__main__":

    with open("imagenet_class_index.json") as f:
        class_dictionary = json.load(f)

    img = cv2.imread("cat2.jpg")

    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = convert_to_intpu_tensor(img)
    tensor = tensor.unsqueeze(0)
    tensor = tensor.to("cuda:0")

    vgg = models.vgg16(pretrained=True)
    vgg = nn.Sequential(
        vgg,
        nn.Softmax(dim=1)
    ).to("cuda:0")
    # vgg.to("cuda:0")
    print("チェック")
    probability, idx, output = forward(vgg, tensor)
    idx = idx[0]
    sort_idx = np.argsort(-output)
    print(output[0, idx])
    print(class_dictionary[str(idx)])
    print(class_dictionary[str(sort_idx[0][0])])
    print("282: tigar_cat", output[0, 282])
    print("チェック終了")

    length = len(range(0, 224, 8))

    tensor_stack = torch.zeros(length * length, 3, 224, 224)

    for i in tqdm(range(0, 224, 8)):
        for j in range(0, 224, 8):
            img_copy = img.copy()
            img_copy[i:i + 64, j:j + 64, :] = 128
            if i == 0 and j == 0:
                cv2.imwrite("huga.jpg", img_copy)
            tensor_stack[int(i / 8) * length + int(j / 8), :, :, :] = convert_to_intpu_tensor(img_copy)

    tensor_stack = tensor_stack.to("cuda:0")

    forward(vgg, tensor_stack[0:10])
    forward(vgg, tensor_stack[10:20])
    forward(vgg, tensor_stack[20:28])

    # forward(vgg, tensor_stack[0:1])
    # forward(vgg, tensor_stack[1:2])
    # forward(vgg, tensor_stack[2:3])
