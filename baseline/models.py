import torch.nn as nn
from torchvision import models


def _get_classification_model(n_classes, name, pretrained=True):
    model_ft = None
    if name == "resnet18":
        model_ft = models.resnet18(pretrained=pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, n_classes)
    elif name == "resnet34":
        model_ft = models.resnet34(pretrained=pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, n_classes)
    elif name == "resnet50":
        model_ft = models.resnet50(pretrained=pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, n_classes)
    elif name == "resnet101":
        model_ft = models.resnet101(pretrained=pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, n_classes)
    elif name == "resnet152":
        model_ft = models.resnet152(pretrained=pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, n_classes)
    return model_ft
