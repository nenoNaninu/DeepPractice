# -*- using:utf-8 -*-
from vgg16_feature import VGG16_4096
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
import torch
import os


def KMeans_cluster(feature, label, n_clusters):
    k_means = KMeans(n_clusters=n_clusters)
    k_means_result = k_means.fit(feature)

    sum = np.zeros((10))
    for i in range(n_clusters):
        idxes = k_means_result.labels_ == i
        ans = label[idxes]
        sum[0] = np.sum(ans == 0) / len(ans)
        sum[1] = np.sum(ans == 1) / len(ans)
        sum[2] = np.sum(ans == 2) / len(ans)
        sum[3] = np.sum(ans == 3) / len(ans)
        sum[4] = np.sum(ans == 4) / len(ans)
        sum[5] = np.sum(ans == 5) / len(ans)
        sum[6] = np.sum(ans == 6) / len(ans)
        sum[7] = np.sum(ans == 7) / len(ans)
        sum[8] = np.sum(ans == 8) / len(ans)
        sum[9] = np.sum(ans == 9) / len(ans)
        print("k_means label{0}".format(i))
        print(
            "0: {0:.2}, 1: {1:.2}, 2: {2:.2}, 3: {3:.2}, 4: {4:.2}, 5: {5:.2}, 6: {6:.2}, 7: {7:.2}, 8: {8:.2}, 9: {9:.2}".format(
                sum[0], sum[1], sum[2], sum[3], sum[4], sum[5], sum[6], sum[7], sum[8], sum[9]))
        print("Max idx is {0}, rate is {1:.2}".format(
            np.argmax(sum), sum[np.argmax(sum)]))


if __name__ == "__main__":
    # DataLoaderを生成
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    home_dir = os.environ["HOME"]

    uecfood10_dir = os.path.join(home_dir, "UECFOOD10")

    image_dataset = ImageFolder(
        # "../../dataset/",
        uecfood10_dir,
        transform)

    trainloader = DataLoader(image_dataset, batch_size=100,
                             shuffle=True, num_workers=2)

    vgg_model = VGG16_4096()
    vgg_model.to("cuda:0")
    vgg_model.eval()

    feature_buffer = np.zeros((4096, 4096))
    idx = 0
    label_buffer = np.empty((0))
    with torch.no_grad():
        for epoch in range(3):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, labels = data
                label_buffer = np.append(label_buffer, labels.detach().numpy())
                inputs = inputs.to("cuda:0")
                num = inputs.shape[0]
                output = vgg_model(inputs)
                output = output.detach().cpu().numpy()
                if 4096 <= idx + num:
                    diff = 4096 - idx
                    feature_buffer[idx:idx + num, :] = output[0:diff]
                    break
                feature_buffer[idx:idx + num, :] = output[0:]
                idx += num
    print("extructe feature")
    label_buffer = label_buffer[:4096]

    print("===original====")
    KMeans_cluster(feature_buffer, label_buffer, 10)

    print("===originalx0.95 pca===")
    pca = PCA(n_components=int(4096 * 0.95))
    pca.fit(feature_buffer)
    transformed = pca.fit_transform(feature_buffer[:4096])
    label_buffer = label_buffer[0:transformed.shape[0]]

    KMeans_cluster(transformed, label_buffer, 10)

    print("===originalx0.9 pca===")
    pca = PCA(n_components=int(4096 * 0.9))
    pca.fit(feature_buffer)
    transformed = pca.fit_transform(feature_buffer[:4096])
    label_buffer = label_buffer[0:transformed.shape[0]]
    KMeans_cluster(transformed, label_buffer, 10)

    print("===128 pca===")
    pca = PCA(n_components=128)
    pca.fit(feature_buffer)
    transformed = pca.fit_transform(feature_buffer[:1500])
    label_buffer = label_buffer[0:transformed.shape[0]]
    KMeans_cluster(transformed, label_buffer, 10)
