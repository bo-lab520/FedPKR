import torch
from torch import nn
import torch.nn.functional as F

from models.ResNetv1 import resnet18, resnet34, resnet50, resnet101, resnet152
from models.ResNetv2 import resnet8, resnet14, resnet20, resnet32, resnet44, resnet56, \
    resnet110, resnet116, resnet8x4, resnet32x4
from models.CNNMnist import cnnmnist, LeNet, SimpleCNN


def get_model(name):
    if name == "resnet18":
        model = resnet18()
    elif name == "resnet34":
        model = resnet34()
    elif name == "resnet50":
        model = resnet50()
    elif name == "resnet101":
        model = resnet101()
    elif name == "resnet152":
        model = resnet152()
    elif name == "resnet8":
        model = resnet8()
    elif name == "resnet14":
        model = resnet14()
    elif name == "resnet20":
        model = resnet20()
    elif name == "resnet32":
        model = resnet32()
    elif name == "resnet44":
        model = resnet44()
    elif name == "resnet56":
        model = resnet56()
    elif name == "resnet110":
        model = resnet110()
    elif name == "resnet116":
        model = resnet116()
    elif name == "resnet8x4":
        model = resnet8x4()
    elif name == "resnet32x4":
        model = resnet32x4()
    elif name == "cnnmnist":
        model = cnnmnist()
    elif name == "lenet":
        model = LeNet()
    elif name == "simple-cnn":
        model = SimpleCNN()

    if torch.cuda.is_available():
        return model.cuda()
    else:
        return model
