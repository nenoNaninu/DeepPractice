# -*- using:utf-8 -*-
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np
import json
from tqdm import tqdm
import matplotlib.pyplot as plt


def forward(model, img):
    model.eval()
    with  torch.no_grad():
        output = model(img)

        idx = torch.max(output, 1)[1]
        probability = output[:, idx]
        probability = probability.cpu().numpy()
        idx = idx.cpu().numpy()
        output = output.cpu().numpy()
    return probability, idx, output


def convert_to_intpu_tensor(img):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    return preprocess(img)


def zeiler_fergus(img_path):
    with open("imagenet_class_index.json") as f:
        class_dictionary = json.load(f)

    img = cv2.imread(img_path)

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

    probability, idx, output = forward(vgg, tensor)
    answer_idx = idx[0]
    probability = probability[0][0]
    print(probability)
    print(output[0, answer_idx])
    print(class_dictionary[str(answer_idx)])

    length = len(range(0, 224, 8))

    tensor_stack = torch.zeros(length * length, 3, 224, 224)

    for i in tqdm(range(0, 224, 8)):
        for j in range(0, 224, 8):
            img_copy = img.copy()
            x_start = 0 if j - 32 < 0 else j - 32
            y_start = 0 if i - 32 < 0 else i - 32

            img_copy[y_start:y_start + 64, x_start:x_start + 64, :] = 128
            if i == 0 and j == 0:
                cv2.imwrite("huga.jpg", img_copy)
            tensor_stack[int(i / 8) * length + int(j / 8), :, :, :] = convert_to_intpu_tensor(img_copy)

    tensor_stack = tensor_stack.to("cuda:0")
    print(tensor_stack.shape)

    output = []
    for i in tqdm(range(0, length * length, 10)):
        _, _, output1 = forward(vgg, tensor_stack[i:i + 10])
        output1 = output1[:, answer_idx]
        output = np.concatenate((output, output1), axis=0)

    print("output shape:", output.shape)

    output = np.reshape(output, (length, length))
    diff = np.abs(probability - output)

    return img, output, diff


if __name__ == "__main__":
    img, output, diff = zeiler_fergus("cat3.jpg")

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.resize(img, (output.shape[0], output.shape[0])))
    plt.title("original")

    plt.subplot(1, 3, 2)
    plt.imshow(output)
    plt.title("output")

    plt.subplot(1, 3, 3)
    plt.imshow(diff)
    plt.title("diff")

    plt.show()
