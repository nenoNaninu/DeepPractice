# -*- using:utf-8 -*-
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn, optim
import torch
from statistics import mean
from tqdm import tqdm
from torchvision.utils import save_image
import os

class ConditionalGenerator(nn.Module):
    def __init__(self,class_num):
        super(ConditionalGenerator, self).__init__()
        self.class_num = class_num
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100+class_num, 32 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(32 * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32 * 8, 32 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32 * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32 * 4, 32 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32 * 2),
            nn.ConvTranspose2d(32 * 2, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x, labels):
        label_tensor = torch.zeros(x.shape[0], self.class_num, 1, 1)
        
        for idx, label in enumerate(labels):
            label_tensor[idx, label, :, :] = 1

        label_tensor = label_tensor.to("cuda:0")
        print("generate")
        print(x.shape)
        print(labels.shape)
        x = torch.cat((x, label_tensor),dim=2)
        print(x.shape)

        x = self.main(x)
        return x

class ConditionalDiscriminator(nn.Module):
    def __init__(self, class_num):
        super(ConditionalDiscriminator, self).__init__()
        self.class_num = class_num
        self.main = nn.Sequential(
            nn.Conv2d(3 + class_num, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32 * 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(32 * 4),
            nn.Conv2d(32 * 4, 32 * 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(32 * 8),
            nn.Conv2d(32 * 8, 1, 4, 1, 0, bias=False)
        )

    def forward(self, x,labels):
        label_tensor = torch.zeros(x.shape[0], self.class_num, 64, 64)
        
        for idx, label in enumerate(labels):
            label_tensor[idx, label, :, :] = 1

        label_tensor = label_tensor.to("cuda:0")
        print("discrimiante")
        print(x.shape)
        print(label.shape)
        print(label_tensor)
        x = torch.cat((x, label_tensor),dim=2)
        x = self.main(x)
        return x.squeeze()


def train_dcgan(g, d, opt_g, opt_d, loader, batch_size):
    log_loss_g = []
    log_loss_d = []

    ones = torch.ones(batch_size).to("cuda:0")
    zeros = torch.zeros(batch_size).to("cuda:0")

    loss_function = nn.BCEWithLogitsLoss()

    print("start training")
    for real_img, labels in tqdm(loader):
        print(real_img.shape)
        print(labels.shape)
        bach_len = len(real_img)

        real_img = real_img.to("cuda:0")

        z = torch.randn(bach_len, 10, 1, 1).to("cuda:0")
        fake_image = g(z, labels)
        fake_image_tensor = fake_image.detach()

        out = d(fake_image, labels)
        loss_g = loss_function(out, ones[:bach_len])
        log_loss_g.append(loss_g.item())

        d.zero_grad()
        g.zero_grad()

        loss_g.backward()

        opt_g.step()

        real_out = d(real_img, labels)

        loss_d_real = loss_function(real_out, ones[:bach_len])

        fake_image = fake_image_tensor

        fake_out = d(fake_image, labels)

        loss_d_fake = loss_function(fake_out, zeros[:bach_len])

        loss_d = loss_d_fake + loss_d_real

        log_loss_d.append(loss_d.item())

        d.zero_grad()
        g.zero_grad()
        loss_d.backward()

        opt_d.step()

    return mean(log_loss_g), mean(log_loss_d)


if __name__ == "__main__":

    batch_size = 320

    image_dataset = ImageFolder(
        # "../../dataset/",
        "../../../UECFOOD10/",
        transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
        ]))

    train_loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=True)

    generator = ConditionalGenerator(10)
    discriminator = ConditionalDiscriminator(10)
    monitor_noize = torch.randn(batch_size, 100, 1, 1)

    # concat用のラベルを作成

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     generator = nn.DataParallel(generator)
    #     discriminator = nn.DataParallel(discriminator)

    generator.to(device)
    discriminator.to(device)

    monitor_noize = monitor_noize.to(device)

    opt_discriminator = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_generator = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    if not os.path.exists('~/space/kadai/kadai5/cgan_save'):
        os.makedirs('~/space/kadai/kadai5/cgan_save')

    if not os.path.exists('~/space/kadai/kadai5/cgan_output_image'):
        os.makedirs('~/space/kadai/kadai5/cgan_output_image')

    save_path = os.path.abspath('~/space/kadai/kadai5/cgan_save')
    output_image_path = os.path.abspath('~/space/kadai/kadai5/cgan_output_image')

    for epoch in range(300):
        train_dcgan(generator, discriminator, opt_generator, opt_discriminator, train_loader, batch_size)

        if epoch % 10 == 0:
            torch.save(generator.state_dict(), "{0}/g_{1:03}.prm".format(save_path,epoch), pickle_protocol=4)
            torch.save(discriminator.state_dict(), "{0}/d_{1:03}.prm".format(save_path,epoch), pickle_protocol=4)

            generated_img = generator(monitor_noize)
            save_image(generated_img, "{0}/{1:03d}.jpg".format(output_image_path, epoch))
