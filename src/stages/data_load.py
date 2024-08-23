import argparse
from typing import Text
import yaml
import os
from pathlib import Path
import urllib.request
import zipfile
from src.utils.logs import get_logger


def data_load(config_path: Text) -> None:
    """Load raw data.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger('DATA_LOAD', log_level=config['base']['log_level'])

    dataset_url=config['data_load']['dataset_url']
    dataset_filename=config['data_load']['dataset_filename']
    dataset_folder=config['data_load']['dataset_folder']
    data_folder = os.path.dirname(dataset_folder)
    dataset_path = os.path.join(data_folder, dataset_filename)

    logger.info('Get dataset')

    Path(data_folder).mkdir(parents=True, exist_ok=True)

    urllib.request.urlretrieve(dataset_url, dataset_path)

    logger.info('Save data')

    with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
        zip_ref.extractall(data_folder)

    Path.unlink(dataset_path)

if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    data_load(config_path=args.config)
