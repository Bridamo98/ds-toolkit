import logging
import os

from ds_toolkit import data_cleaning
from ds_toolkit.utils.types import DataCleaningConfigFile


def test_data_cleaning():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file_path = os.path.join(script_dir, "data-cleaning.yml")
    # Read config file ----------------------------------------------------
    dt_config = data_cleaning.read_dt_config_file(config_file_path)
    handled_dt_config: DataCleaningConfigFile = data_cleaning.handle_dt_config(
        dt_config
    )
    # pprint(handled_dt_config.get(), sort_dicts=False)

    # Read original dataset -----------------------------------------------
    kwargs = {"params": os.path.join(script_dir, handled_dt_config.general.input)}
    original_dataset = data_cleaning.read_dataset(**kwargs)
    logging.warning(original_dataset)

    # 1. PREPARE VARIABLE NAMES
    t1 = data_cleaning.test(
        params={
            "params": handled_dt_config.stages.prepare.variable_names
            if handled_dt_config.stages.prepare is not None
            else None
        },
        transformation=data_cleaning.prepare_var_names,
        original_dataset=original_dataset,
        d_is_returned=True,
        transformation_verbose=True,
    )

    # 2. PREPARE VARIABLE VALUES
    t2 = data_cleaning.test(
        params={
            "params": handled_dt_config.stages.prepare.variable_values
            if handled_dt_config.stages.prepare is not None
            else None
        },
        transformation=data_cleaning.prepare_var_values,
        original_dataset=t1,
        d_is_returned=True,
        transformation_verbose=True,
    )

    # 3. HANDLE DUPLICATE DATA
    t3 = data_cleaning.test(
        params={"params": handled_dt_config.stages.remove_duplicate},
        transformation=data_cleaning.handle_duplicate_data,
        original_dataset=t2,
        d_is_returned=True,
        transformation_verbose=True,
    )

    # 4. HANDLE IRRELEVANT DATA
    t4 = data_cleaning.test(
        params={"params": handled_dt_config.stages.remove_irrelevant},
        transformation=data_cleaning.handle_irrelevant_data,
        original_dataset=t3,
        d_is_returned=True,
        transformation_verbose=True,
    )

    # 4. HANDLE IRRELEVANT DATA
    t5 = data_cleaning.test(
        params={"params": handled_dt_config.stages.handle_missing_data},
        transformation=data_cleaning.handle_missing_data,
        original_dataset=t4,
        transformation_verbose=True,
        d_is_returned=True,
    )

    # 4. HANDLE IRRELEVANT DATA
    data_cleaning.test(
        params={"params": handled_dt_config.stages.handle_outliers},
        transformation=data_cleaning.handle_outliers,
        original_dataset=t5,
        transformation_verbose=True,
        test_verbose=True,
    )
