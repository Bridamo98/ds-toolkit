import os

from ds_toolkit import data_cleaning


def test_data_cleaning():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    relative_config_path = "assets/data_cleaning.yml"
    data_cleaning.clean(root_dir, relative_config_path, return_intermediate_steps=True)
