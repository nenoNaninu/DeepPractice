# -*- using:utf-8 -*-
import time
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm
from sklearn.cluster import KMeans
import numpy as np
from torchvision.utils import save_image
import os


# 入力画像は32x32x3 出力は256x1x1
class Encorder(nn.Module):
    def __init__(self):
        super(Encorder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2))

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(4, 4))

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2))

    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        return output


class Decorder(nn.Module):
    def __init__(self):
        super(Decorder, self).__init__()
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))

        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True))

    def forward(self, input):
        output = self.deconv1(input)
        output = self.deconv2(output)
        output = self.deconv3(output)
        output = self.deconv4(output)
        output = self.deconv5(output)
        return output


def add_noise(img):
    noise = torch.randn(img.size()) * 0.2
    noisy_img = img + noise
    return noisy_img


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    encorder = Encorder()
    decorder = Decorder()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        encorder = nn.DataParallel(encorder)
        decorder = nn.DataParallel(decorder)

    encorder.to(device)
    decorder.to(device)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 500 * torch.cuda.device_count()

    trainset = torchvision.datasets.CIFAR10(root='~/space/pytorch_datset', train=True,
                                            download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='~/space/pytorch_datset', train=False,
                                           download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size,
                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    loss_func = nn.MSELoss()
    optimizer_encorder = optim.Adam(encorder.parameters(), lr=0.001, betas=(0.5, 0.999))
    optimizer_decorder = optim.Adam(decorder.parameters(), lr=0.001, betas=(0.5, 0.999))

    if not os.path.exists('./output_image'):
        os.makedirs('./output_image')

    if not os.path.exists('./model_save'):
        os.makedirs('./model_save')

    summaryWriter = SummaryWriter("./tensorborad/log")

    counter = 0
    for epoch in range(1):
        running_loss = 0.0
        print("epoch: ", epoch)
        for i, data in tqdm(enumerate(trainloader, 0)):
            # get the inputs
            original_inputs, labels = data

            inputs = add_noise(original_inputs)

            inputs, labels = inputs.to("cuda:0"), labels.to("cuda:0")

            # zero the parameter gradients
            optimizer_encorder.zero_grad()
            optimizer_decorder.zero_grad()

            # forward + backward + optimize
            outputs = encorder(inputs)
            outputs = decorder(outputs)

            # loss = loss_func(original_inputs.view(original_inputs.shape[0], -1), outputs.view(outputs.shape[0], -1))
            loss = loss_func(outputs, original_inputs.to("cuda:0"))
            loss.backward()

            optimizer_encorder.step()
            optimizer_decorder.step()

            # print statistics
            # running_loss += loss.item()

            summaryWriter.add_scalar('train_loss', loss.item(), counter)
            counter += 1
            if i % 10 == 9:
                save_image(outputs, "./output_image/{:03d}.jpg".format(epoch))

    current_dir = os.getcwd()

    torch.save(encorder.state_dict(), os.path.join(current_dir, "./model_save/encorder.prm"))
    torch.save(decorder.state_dict(), os.path.join(current_dir, "./model_save/decorder.prm"))

    encorder.eval()
    decorder.eval()
    print("=====")
    # その後分類。
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        if i == 1:
            break
        original_inputs, labels = data

        inputs = add_noise(original_inputs)

        inputs = inputs.to("cuda:0")
        labels = labels.cpu().numpy()

        with torch.no_grad():
            outputs = encorder(inputs)
            np_array = outputs.cpu().numpy()
            np_array = np.reshape(np_array, [np_array.shape[0], -1])

            k_means = KMeans(n_clusters=10)
            k_means_result = k_means.fit(np_array)

            for i in range(10):
                idxes = k_means_result.labels_ == i
                ans = labels[idxes]
                sum_0 = np.sum(ans == 0) / len(ans)
                sum_1 = np.sum(ans == 1) / len(ans)
                sum_2 = np.sum(ans == 2) / len(ans)
                sum_3 = np.sum(ans == 3) / len(ans)
                sum_4 = np.sum(ans == 4) / len(ans)
                sum_5 = np.sum(ans == 5) / len(ans)
                sum_6 = np.sum(ans == 6) / len(ans)
                sum_7 = np.sum(ans == 7) / len(ans)
                sum_8 = np.sum(ans == 8) / len(ans)
                sum_9 = np.sum(ans == 9) / len(ans)
                print("k_means label{0}".format(i))
                print(
                    "0: {0:.2}, 1: {1:.2}, 2: {2:.2}, 3: {3:.2}, 4: {4:.2}, 5: {5:.2}, 6: {6:.2}, 7: {7:.2}, 8: {8:.2}, 9: {9:.2}".format(
                        sum_0, sum_1, sum_2,
                        sum_3, sum_4, sum_5,
                        sum_6, sum_7, sum_8,
                        sum_9))

        # count = np.sum(result.labels_ == labels)
        # print("reate: ", count / len(labels))
        # break
