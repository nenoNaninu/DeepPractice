# -*- using:utf-8 -*-
import torch
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np
from VanilaBackprop import VanillaBackprop


def convert_to_grayscale(im_as_arr):
    """
        Converts 3d image to grayscale

    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)

    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im


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


def back_prop(img_path):
    img = cv2.imread(img_path)

    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = convert_to_intpu_tensor(img)
    tensor = tensor.unsqueeze(0)
    tensor.requires_grad = True

    tensor = tensor.to("cuda:0")
    print(tensor.requires_grad)

    vgg = models.vgg16(pretrained=True).to("cuda:0")
    vgg.eval()

    output = vgg(tensor)

    idx = torch.max(output, 1)[1]
    print(idx)
    vgg.zero_grad()

    vanila_backprop = VanillaBackprop(vgg)

    vanilla_grads = vanila_backprop.generate_gradients(tensor, idx)

    grayscale_vanilla_grads = convert_to_grayscale(vanilla_grads)
    # Save grayscale gradients
    print("aaa", grayscale_vanilla_grads.shape)
    print(type(grayscale_vanilla_grads))
    grayscale_vanilla_grads = grayscale_vanilla_grads.transpose(1, 2, 0)
    print(grayscale_vanilla_grads.shape)
    print(grayscale_vanilla_grads.dtype)
    cv2.imwrite("gray.jpg", grayscale_vanilla_grads * 255)


if __name__ == "__main__":
    back_prop("cat3.jpg")
