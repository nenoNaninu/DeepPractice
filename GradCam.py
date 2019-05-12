# -*- using:utf-8 -*-
# original code(MIT) -> https://github.com/utkuozbulak/pytorch-cnn-visualizations/blob/master/src/gradcam.py
# https://github.com/utkuozbulak/pytorch-cnn-visualizations/blob/master/LICENSE

from PIL import Image
import numpy as np
import torch
import cv2
import torchvision.transforms as transforms
from torchvision import models
import matplotlib.pyplot as plt
from misc_functions import save_class_activation_images


class CamExtractor:
    """
        Extracts cam features from the model
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad
        print(self.gradients.shape)

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
            この関数が返すのは(所望のConv層の出力, model.featuresの最終出力
        """
        conv_output = None
        for module_pos, module in self.model.features._modules.items():
            # モデル内のモジュールをひとつひとつ取り出してフォワーディングする。
            x = module(x)  # Forward
            # 引張だしたい該当するモジュールに引っ掛ける
            if int(module_pos) == self.target_layer:
                print(type(x))
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer(所望のレイヤの出力は保存しておく。

        return conv_output, x

    def forward_pass(self, x):
        """
            外部から呼ばれるのはこっち。
            xは入力画像
            返り値は
            (ターゲットレイヤのConvの出力,最終レイヤ(classfier)の出力
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = self.model.classifier(x)
        return conv_output, x


class GradCam:
    """
        Produces class activation map
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # メンバとしてエクストラクタを持つ。
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())

        # Target for backprop
        # one hotベクトル作成
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        one_hot_output = one_hot_output.to("cuda:0")

        # Zero grads
        self.model.features.zero_grad()
        self.model.classifier.zero_grad()

        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)

        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.cpu().numpy()[0]
        # Get convolution outputs

        #  勾配こみこみTensorはそのままcpu()できなくて、dataとかつかって中身いちいち取り出さないといけない
        # (512,14,14とか)
        target = conv_output.data.cpu().numpy()[0]

        # Get weights from gradients
        # グローバルアベレージプーリング
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient

        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                                                    input_image.shape[3]), Image.ANTIALIAS)) / 255
        # ^ I am extremely unhappy with this line. Originally resizing was done in cv2 which
        # supports resizing numpy matrices with antialiasing, however,
        # when I moved the repository to PIL, this option was out of the window.
        # So, in order to use resizing with ANTIALIAS feature of PIL,
        # I briefly convert matrix to PIL image and then back.
        # If there is a more beautiful way, do not hesitate to send a PR.
        return cam


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


if __name__ == '__main__':
    img = cv2.imread("cat3.jpg")

    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = convert_to_intpu_tensor(img)
    tensor = tensor.unsqueeze(0)
    tensor.requires_grad = True

    tensor = tensor.to("cuda:0")

    vgg = models.vgg16(pretrained=True).to("cuda:0")
    vgg.eval()

    output = vgg(tensor)

    idx = torch.max(output, 1)[1]
    vgg.zero_grad()

    print(vgg)
    # # Grad cam
    grad_cam = GradCam(vgg, target_layer=29)
    # # Generate cam mask
    cam = grad_cam.generate_cam(tensor, idx)

    plt.imshow(cam)
    plt.show()

    # # Save mask
    original_image = Image.fromarray(np.uint8(img))
    save_class_activation_images(original_image, cam, "cat3")
    print('Grad cam completed')
