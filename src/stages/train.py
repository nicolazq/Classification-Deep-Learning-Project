import argparse
import os
from pathlib import Path
from typing import Text

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from src.utils.dataloaders import get_dataloaders
from src.utils.initialize_model import initialize_model
from src.utils.logs import get_logger
from src.utils.train_model import train_model

import mlflow
from dotenv import find_dotenv, load_dotenv

def train(config_path: Text) -> None:

    _ = load_dotenv(find_dotenv())

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger("TRAIN", log_level=config["base"]["log_level"])

    experiment_name = config["base"]["mlflow_experiment_name"]
    mlflow.set_experiment(experiment_name)

    random_state = config["base"]["random_state"]
    torch.manual_seed(random_state)

    logger.info("initizalize_model")

    model_name = config["train"]["base_model_name"]
    model_trainable = config["train"]["base_model_name"]
    model = initialize_model(model_name, model_trainable)

    data_folder = config["data_load"]["dataset_folder"]
    dataloaders, _ = get_dataloaders(data_folder, random_state)

    criterion = nn.CrossEntropyLoss()

    mlflow.enable_system_metrics_logging()
    with mlflow.start_run():

        lr = config["train"]["optim"]["SGD"]["lr"]
        momentum = config["train"]["optim"]["SGD"]["momentum"]

        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

        step_size = config["train"]["lrs"]["StepLR"]["step_size"]
        gamma = config["train"]["lrs"]["StepLR"]["gamma"]
        # Decay LR by a factor of _gamma_ every _step_size_ epochs
        exp_lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )

        num_epochs = config["train"]["num_epochs"]

        params = {
            "optimizer": "optimizer",
            "lr": lr,
            "momentum":momentum,
            "step_size":step_size,
            "gamma":gamma,
            "num_epochs":num_epochs,
        }
        mlflow.log_params(params)

        logger.info("train model")
        model = train_model(
            model,
            dataloaders,
            criterion,
            optimizer,
            exp_lr_scheduler,
            num_epochs=num_epochs,
        )

    model_path = config["train"]["model_path"]
    model_folder = os.path.dirname(model_path)
    Path(model_folder).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)

    logger.info("model saved")


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    train(config_path=args.config)
