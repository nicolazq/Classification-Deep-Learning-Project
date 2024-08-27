import argparse
import json
from pathlib import Path
from typing import Dict, Text

import joblib
import torch
import yaml
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, f1_score

from src.report.visualize import plot_confusion_matrix
from src.utils.dataloaders import get_dataloaders
from src.utils.initialize_model import initialize_model
from src.utils.logs import get_logger

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_prediction(model, dataloaders):
    was_training = model.training
    model.eval()

    labels_all = []
    preds_all = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders["val"]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            labels_all.extend(labels.tolist())
            preds_all.extend(preds.tolist())

        model.train(mode=was_training)

    return labels_all, preds_all


def evaluate(config_path: Text) -> None:
    """Evaluate model.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger("EVALUATE", log_level=config["base"]["log_level"])
    logger.info("Load model")

    model_name = config["train"]["base_model_name"]
    model = initialize_model(model_name, False)

    model_path = config["train"]["model_path"]
    model.load_state_dict(torch.load(model_path))

    _ = model.eval()

    data_folder = config["data_load"]["dataset_folder"]
    random_state = config["base"]["random_state"]
    dataloaders, target_names = get_dataloaders(data_folder, random_state)

    y_test, prediction = get_prediction(model, dataloaders)

    cm = confusion_matrix(y_test, prediction)
    print(cm)

    cm_plot = plot_confusion_matrix(cm, target_names, normalize=False)

    f1 = f1_score(y_true=y_test, y_pred=prediction, average="macro")
    logger.info(f"f1 = {f1}")

    reports_folder = Path(config["evaluate"]["reports_dir"])
    Path(reports_folder).mkdir(parents=True, exist_ok=True)

    confusion_matrix_png_path = (
        reports_folder / config["evaluate"]["confusion_matrix_image"]
    )
    cm_plot.savefig(confusion_matrix_png_path)

    metrics_path = reports_folder / config["evaluate"]["metrics_file"]
    metrics = {"f1": f1}
    with open(metrics_path, "w") as mf:
        json.dump(obj=metrics, fp=mf, indent=4)


def convert_to_labels(indexes, labels):
    result = []
    for i in indexes:
        result.append(labels[i])
    return result


def write_confusion_matrix_data(y_true, predicted, labels, filename):
    assert len(predicted) == len(y_true)
    predicted_labels = convert_to_labels(predicted, labels)
    true_labels = convert_to_labels(y_true, labels)
    cf = pd.DataFrame(
        list(zip(true_labels, predicted_labels)), columns=["y_true", "predicted"]
    )
    cf.to_csv(filename, index=False)


def evaluate_model(config_path: Text) -> None:
    """Evaluate model.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger("EVALUATE", log_level=config["base"]["log_level"])

    logger.info("Load model")
    model_path = config["train"]["model_path"]
    model = joblib.load(model_path)

    logger.info("Load test dataset")
    test_df = pd.read_csv(config["data_split"]["testset_path"])

    logger.info("Evaluate (build report)")
    target_column = config["featurize"]["target_column"]
    y_test = test_df.loc[:, target_column].values
    X_test = test_df.drop(target_column, axis=1).values

    prediction = model.predict(X_test)
    f1 = f1_score(y_true=y_test, y_pred=prediction, average="macro")

    labels = load_iris(as_frame=True).target_names.tolist()
    cm = confusion_matrix(y_test, prediction)

    report = {"f1": f1, "cm": cm, "actual": y_test, "predicted": prediction}

    logger.info("Save metrics")
    # save f1 metrics file
    reports_folder = Path(config["evaluate"]["reports_dir"])
    metrics_path = reports_folder / config["evaluate"]["metrics_file"]

    json.dump(obj={"f1_score": report["f1"]}, fp=open(metrics_path, "w"))

    logger.info(f"F1 metrics file saved to : {metrics_path}")

    logger.info("Save confusion matrix")
    # save confusion_matrix.png
    plt = plot_confusion_matrix(cm=report["cm"], target_names=labels, normalize=False)
    confusion_matrix_png_path = (
        reports_folder / config["evaluate"]["confusion_matrix_image"]
    )
    plt.savefig(confusion_matrix_png_path)
    logger.info(f"Confusion matrix saved to : {confusion_matrix_png_path}")

    confusion_matrix_data_path = (
        reports_folder / config["evaluate"]["confusion_matrix_data"]
    )
    write_confusion_matrix_data(
        y_test, prediction, labels=labels, filename=confusion_matrix_data_path
    )
    logger.info(f"Confusion matrix data saved to : {confusion_matrix_data_path}")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    evaluate(config_path=args.config)
