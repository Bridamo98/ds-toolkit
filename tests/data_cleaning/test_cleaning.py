import csv
import os

from pandas import DataFrame

from ds_toolkit.data_cleaning import clean, save_result


def are_csv_files_equal(file1: str, file2: str):
    with open(file1, "r") as csv_file1, open(file2, "r") as csv_file2:
        reader1 = csv.reader(csv_file1)
        reader2 = csv.reader(csv_file2)

        # Compare each row in the CSV files
        for row1, row2 in zip(reader1, reader2):
            assert row1 == row2

        # Check if both files have the same number of rows
        assert len(list(reader1)) == len(list(reader2))


def prepare_test(root_dir: str):
    relative_config_path = "assets/data_cleaning.yml"
    update_mocked_data = False  # Set to True if mocked data is necessary to update
    intermediate_steps = clean(
        root_dir,
        relative_config_path,
        return_intermediate_steps=True,
        save_intermediate_steps=update_mocked_data,
        steps_dir="assets/mock/expected/",
        transformation_verbose=True,
    )

    if intermediate_steps is not None:
        for step in intermediate_steps:
            save_result(
                DataFrame(step["dataframe"]),
                os.path.join(root_dir, f"assets/mock/obtained/{str(step['name'])}.csv"),
            )

    return intermediate_steps


def test_data_cleaning():
    root_dir = os.path.dirname(os.path.abspath(__file__))

    intermediate_steps = prepare_test(root_dir)

    if intermediate_steps is not None:
        for step in intermediate_steps:
            are_csv_files_equal(
                os.path.join(root_dir, f"assets/mock/expected/{str(step['name'])}.csv"),
                os.path.join(root_dir, f"assets/mock/obtained/{str(step['name'])}.csv"),
            )
