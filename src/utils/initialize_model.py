import torch.nn as nn
from torchvision import models


def initialize_model(model_name, base_model_trainable):

    num_classes = 2
    model = None

    if model_name == "resnet18":
        model = models.resnet18(weights="IMAGENET1K_V1")

    else:
        print("Pick a valid model name")
        exit()

    if base_model_trainable == False:
        for param in model.parameters():
            param.requires_grad = False

    # requires_grad=True by default
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model
