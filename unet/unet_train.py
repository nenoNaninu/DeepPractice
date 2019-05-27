# -*- using:utf-8 -*-
import torch
from torch import nn, optim
from unet import UNet
from pascal_dataset import PascalVOC
from torch.utils.data import DataLoader
import os
from tensorboardX import SummaryWriter
from tqdm import tqdm


if __name__ == "__main__":

    img_list_path = "/export/space0/shimoda-k/wseg/data/img_list.txt"
    mask_list_path = "/export/space0/shimoda-k/wseg/data/mask_list.txt"
    summaryWriter_log_dir = "./tensorborad/unet/log"
    dir_checkpoint = os.path.join(os.getcwd(), "unet_train/")

    if not os.path.exists(summaryWriter_log_dir):
        os.makedirs(summaryWriter_log_dir)

    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint)

    pascalVOC = PascalVOC(img_list_path, mask_list_path, img_size=224)

    batch_size = 50
    trainloader = DataLoader(
        pascalVOC, batch_size=batch_size, shuffle=True, num_workers=6)
    print("batch num", len(trainloader))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = UNet(3, 20)

    summaryWriter = SummaryWriter(summaryWriter_log_dir)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)

    net.to(device)

    lr = 0.1
    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)

    criterion = nn.BCELoss()

    epochs = 250

    for epoch in range(epochs):
        print("epoch: ", epoch)
        net.train()
        epoch_loss = 0

        for i, data in tqdm(enumerate(trainloader)):
            img, mask = data
            img = img.to(device)
            true_mask = mask.to(device)

            predict_mask = net(img)
            # print(predict_mask.dtype)
            # print(true_mask.dtype)
            loss = criterion(predict_mask, true_mask)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save(net.state_dict(),
                   dir_checkpoint + 'CP{}.prm'.format(epoch + 1))
        print("epoch loss is ", epoch_loss)
        print('Checkpoint {} saved !'.format(epoch + 1))
        summaryWriter.add_scalar('train_loss', epoch_loss, epoch)
