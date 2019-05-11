#-*- using:utf-8 -*-
import time
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim


# 適当に畳み込みしていく。
class Cifer10Net(nn.Module):
    def __init__(self, num_classes):
        super(Cifer10Net, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True))

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True))

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 64, 1, 1, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True))

        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True))

        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True))

        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 128, 1, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True))

        self.conv8 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2))

        self.conv9 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True))

        self.conv10 = nn.Sequential(
            nn.Conv2d(64, 32, 1, 1, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True))

        self.fc = nn.Linear(32 * 16 * 16, num_classes)

    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.conv6(output)
        output = self.conv7(output)
        output = self.conv8(output)
        output = self.conv9(output)
        output = self.conv10(output)

        output = output.view(-1, 32 * 16 * 16)

        output = self.fc(output)

        return output


def shape_test():
    dummy = torch.zeros(1, 3, 32, 32).to("cuda:0")
    cifer10Net = Cifer10Net(10).to("cuda:0")
    output = cifer10Net(dummy)
    print(output.shape)


if __name__ == '__main__':
    # shape_test()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cifer10Net = Cifer10Net(10).to("cuda:0")

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        cifer10Net = nn.DataParallel(cifer10Net)

    cifer10Net.to(device)

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

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cifer10Net.parameters(), lr=0.001, momentum=0.9)

    start = time.time()

    for epoch in range(20):  
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to("cuda:0"), labels.to("cuda:0")

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = cifer10Net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

    print('Finished Training')
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    with torch.no_grad():
        cifer10Net.eval()
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to("cuda:0"), labels.to("cuda:0")
            outputs = cifer10Net(inputs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(batch_size):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print(class_correct)
    print(class_total)
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))