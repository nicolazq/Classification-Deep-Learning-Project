import os

import torch
from torchvision import datasets, models, transforms


def get_dataloaders(data_folder, random_state):

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_folder, x), data_transforms[x])
        for x in ["train", "val"]
    }

    class_names = image_datasets["train"].classes

    generator1 = torch.Generator().manual_seed(random_state)

    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x],
            batch_size=4,
            shuffle=True,
            num_workers=4,
            generator=generator1,
        )
        for x in ["train", "val"]
    }

    return dataloaders, class_names
