# Required libraries
import logging
from datetime import datetime
from math import floor
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pandas.core.frame import DataFrame
from utils.types import (
    PRIMITIVES,
    DataCleaningConfigFile,
    FillMethod,
    HandleMissingData,
    HandleOutliers,
    NumericalApplyMethod,
    NumericalDetectMethod,
    RemoveDuplicate,
    RemoveIrrelevant,
    VariableNames,
    VariableNamesApply,
    VariableType,
    VariableValues,
    VariableValuesApplyFunction,
    VariableValuesApplyParams,
)

logging.getLogger().setLevel(logging.WARNING)
FORMAT = "%(message)s"
logging.basicConfig(format=FORMAT)

LEVEL_1 = "    "
LEVEL_2 = "       "
LEVEL_3 = "          "
LEVEL_4 = "             "

SKIP = "⏭  missing params: SKIPPED"
SKIP_TO_NAMES = "⏭  missing to_names: SKIPPED"
SKIP_TO_TYPES = "⏭  missing to_types: SKIPPED"


def setup_verbose(verbose: bool) -> None:
    if verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)


def turnoff_verbose() -> None:
    logging.getLogger().setLevel(logging.WARNING)


def log_dataset(dataset: DataFrame, level: str) -> None:
    str_records = dataset.to_string().split("\n")
    logging.info(f"{level}📊 {str_records[0]}")
    for record in str_records[1:]:
        logging.info(f"{level} ▫ {record}")


# Read the dataset


def read_dataset(params: str) -> DataFrame:
    return pd.read_csv(params)


# Test each transformation


def test_transformation(
    transformation,
    org_dataset: DataFrame,
    d_is_returned: bool,
    test_verbose: bool,
    transformation_verbose: bool,
    **kwargs,
) -> DataFrame | None:
    setup_verbose(test_verbose)

    dataset = org_dataset.copy(deep=True)
    logging.info("BEFORE:")
    logging.info(dataset)

    turnoff_verbose()

    transformation(dataset, transformation_verbose, **kwargs)

    setup_verbose(test_verbose)

    logging.info("AFTER:")
    logging.info(dataset)

    turnoff_verbose()

    if d_is_returned:
        return dataset


def test(
    params: dict[str, Any],
    transformation,
    original_dataset: DataFrame | None,
    d_is_returned: bool = False,
    test_verbose: bool = False,
    transformation_verbose: bool = False,
) -> DataFrame | None:
    kwargs = params
    if original_dataset is None:
        return None
    return test_transformation(
        transformation,
        original_dataset,
        d_is_returned,
        test_verbose,
        transformation_verbose,
        **kwargs,
    )


# Basic utils


def handle_dt_config(dt_config) -> DataCleaningConfigFile:
    return DataCleaningConfigFile(dt_config)


def read_dt_config_file(path: str) -> Any | None:
    with open(path, "r") as stream:
        try:
            config_file = yaml.safe_load(stream)
            return config_file
        except yaml.YAMLError as exc:
            print(exc)


def print_basic_info(dataset: DataFrame) -> None:
    # Print shape
    print(dataset.shape)
    # Print head
    dataset.head()
    # Print basic information
    dataset.info()
    # Print dataset
    print(dataset)


def prepare_var_names(
    dataset: DataFrame, transformation_verbose: bool, params: VariableNames | None
) -> None:
    setup_verbose(transformation_verbose)

    logging.info("(1) PREPARE VARIABLE NAMES:")

    if params is not None:
        if params.apply.value == VariableNamesApply.UPPER:
            dataset.columns = dataset.columns.str.upper()
            logging.info(f"{LEVEL_1}✅ variable names transformed to UPPER")
        elif params.apply.value == VariableNamesApply.LOWER:
            logging.info(f"{LEVEL_1}✅ variable names transformed to LOWER")
            dataset.columns = dataset.columns.str.lower()
    else:
        logging.info(f"{LEVEL_1}{SKIP}")

    turnoff_verbose()


def get_apply_function_value(
    dataset: DataFrame,
    column: str,
    function_to_apply: VariableValuesApplyFunction | FillMethod,
    apply_params: VariableValuesApplyParams | None,
) -> None:
    if isinstance(function_to_apply, VariableValuesApplyFunction):
        if function_to_apply == VariableValuesApplyFunction.INT:
            dataset[column] = dataset[column].apply(lambda x: int(x))

        if function_to_apply == VariableValuesApplyFunction.STR:
            dataset[column] = dataset[column].apply(
                lambda x: str(x) if x is not np.nan else np.nan
            )

        if function_to_apply == VariableValuesApplyFunction.FLOAT:
            dataset[column] = dataset[column].apply(lambda x: int(x))

        if function_to_apply == VariableValuesApplyFunction.BOOL:
            dataset[column] = dataset[column].apply(lambda x: bool(x))

        if function_to_apply == VariableValuesApplyFunction.ROUND:
            if apply_params is not None and apply_params.digits_to_round is not None:
                dataset[column] = dataset[column].apply(
                    lambda x: round(x, apply_params.digits_to_round)
                )

            dataset[column] = dataset[column].apply(lambda x: round(x, 1))

        if function_to_apply == VariableValuesApplyFunction.UPPER:
            dataset[column] = dataset[column].apply(
                lambda x: str(x).upper() if x is not np.nan else np.nan
            )

        if function_to_apply == VariableValuesApplyFunction.LOWER:
            dataset[column] = dataset[column].apply(
                lambda x: str(x).lower() if x is not np.nan else np.nan
            )

        if function_to_apply == VariableValuesApplyFunction.DATETIME:
            dataset[column] = pd.to_datetime(dataset[column])

    if isinstance(function_to_apply, FillMethod):
        if function_to_apply == FillMethod.BFILL:
            dataset[column].fillna(method="bfill", inplace=True)
        if function_to_apply == FillMethod.FFILL:
            dataset[column].fillna(method="ffill", inplace=True)
        if function_to_apply == FillMethod.INTERPOLATE:
            dataset[column] = dataset[column].interpolate()
        if function_to_apply == FillMethod.MEAN:
            dataset[column] = dataset[column].fillna(dataset[column].median())
        if function_to_apply == FillMethod.MODE:
            dataset[column] = dataset[column].fillna(dataset[column].mode()[0])
        if function_to_apply == FillMethod.MEDIAN:
            dataset[column] = dataset[column].fillna(dataset[column].median())
        if function_to_apply == FillMethod.REMOVE_R:
            dataset.dropna(subset=[column], inplace=True, axis=0)


def apply_function_to_columns(
    dataset: DataFrame,
    function_to_apply: VariableValuesApplyFunction | FillMethod | None,
    apply_params: VariableValuesApplyParams | None,
    columns: list[str],
) -> None:
    for column in columns:
        if function_to_apply is not None:
            get_apply_function_value(dataset, column, function_to_apply, apply_params)


def get_pandas_types(types: list[VariableType]) -> list[str]:
    pandas_types = list[str](
        map(lambda x: x.value if x != VariableType.STR else "object", types)
    )
    return pandas_types


def apply_function(
    dataset: DataFrame,
    functions_to_apply: list[VariableValuesApplyFunction] | list[FillMethod],
    apply_params: VariableValuesApplyParams | None,
    to_names: list[str] | None,
    to_types: list[VariableType] | None,
) -> None:
    level_case = LEVEL_2

    if isinstance(functions_to_apply[0], FillMethod):
        level_case = LEVEL_3

    if to_names is not None:
        for function_to_apply in functions_to_apply:
            apply_function_to_columns(
                dataset, function_to_apply, apply_params, to_names
            )

        logging.info(
            f"{level_case}✅ apply {list(map(lambda x: x.value, functions_to_apply))} to {to_names}"
        )

    else:
        logging.info(f"{level_case}{SKIP_TO_NAMES}")

    if to_types is not None:
        pandas_types = get_pandas_types(to_types)
        columns = list(dataset.select_dtypes(include=pandas_types).columns)

        for function_to_apply in functions_to_apply:
            apply_function_to_columns(dataset, function_to_apply, apply_params, columns)

        logging.info(
            f"{level_case}✅ apply {list(map(lambda x: x.value, functions_to_apply))} to types {list(map(lambda x: x.value, to_types))} -> {columns}"
        )

    else:
        logging.info(f"{level_case}{SKIP_TO_TYPES}")


def prepare_var_values(
    dataset: DataFrame,
    transformation_verbose: bool,
    params: list[VariableValues] | None,
) -> None:
    setup_verbose(transformation_verbose)

    logging.info("(2) PREPARE VARIABLE VALUES:")

    if params is not None:
        for index, param in enumerate(params):
            functions_to_apply: list[VariableValuesApplyFunction] = param.apply.function
            apply_params: VariableValuesApplyParams | None = param.apply.params
            to_names: list[str] | None = param.to.variable_names
            to_types: list[VariableType] | None = param.to.variable_types

            logging.info(f"{LEVEL_1}🔹 Transformation #{index + 1}")

            apply_function(
                dataset, functions_to_apply, apply_params, to_names, to_types
            )

    else:
        logging.info(f"{LEVEL_1}{SKIP}")

    turnoff_verbose()


def remove_duplicate_records(dataset: DataFrame) -> DataFrame:
    duplicate_records = dataset[dataset.duplicated(keep="first")]
    dataset.drop_duplicates(keep="first", inplace=True)
    return duplicate_records


def remove_duplicate_variables(dataset: DataFrame) -> list[str]:
    not_duplicate_vars = dataset.T.drop_duplicates().T
    duplicate_vars = [
        col_name
        for col_name in dataset.columns
        if col_name not in not_duplicate_vars.columns
    ]
    dataset.drop(columns=duplicate_vars, inplace=True)
    return list[str](duplicate_vars)


def handle_duplicate_data(
    dataset: DataFrame, transformation_verbose: bool, params: RemoveDuplicate | None
) -> None:
    setup_verbose(transformation_verbose)
    logging.info("(3) HANDLE DUPLICATE DATA:")

    if params is not None:
        logging.info(f"{LEVEL_1}🔹 Remove duplicate records")

        if params.records is not None:
            before_deletion = len(dataset)
            removed_records_ids = list(remove_duplicate_records(dataset).index)
            after_deletion = len(dataset)
            diff = before_deletion - after_deletion
            percent = round((diff / before_deletion) * 100, 2)
            logging.info(
                f"{LEVEL_2}✅ before: {before_deletion} | after: {after_deletion} | diff: {diff} ({percent}%) => ids {removed_records_ids}"
            )
            # log_dataset(removed_records, LEVEL_2)

        else:
            logging.info(f"{LEVEL_2}⏭  records not specified")

        logging.info(f"{LEVEL_1}🔹 Remove duplicate variables")

        if params.variables is not None:
            before_deletion = len(dataset.columns)
            dup_vars = remove_duplicate_variables(dataset)
            after_deletion = len(dataset.columns)
            diff = before_deletion - after_deletion
            percent = round((diff / before_deletion) * 100, 2)
            logging.info(
                f"{LEVEL_2}✅ before: {before_deletion} | after: {after_deletion} | diff: {diff} ({percent}%) => {dup_vars}"
            )
        else:
            logging.info(f"{LEVEL_2}⏭  variables not specified")

    else:
        logging.info(f"{LEVEL_1}{SKIP}")

    turnoff_verbose()


def remove_one_lvl_variables(dataset: DataFrame) -> list[str]:
    one_level_vars = [
        col_name for col_name in dataset.columns if dataset[col_name].nunique() == 1
    ]
    dataset.drop(columns=one_level_vars, inplace=True)
    return list[str](one_level_vars)


def handle_irrelevant_data(
    dataset: DataFrame, transformation_verbose: bool, params: RemoveIrrelevant | None
):
    setup_verbose(transformation_verbose)
    logging.info("(4) HANDLE IRRELEVANT DATA:")

    if params is not None:
        logging.info(f"{LEVEL_1}🔹 Remove one level variables")

        if params.one_level_variables is not None:
            before_deletion = len(dataset.columns)
            dup_vars = remove_one_lvl_variables(dataset)
            after_deletion = len(dataset.columns)
            diff = before_deletion - after_deletion
            percent = round((diff / before_deletion) * 100, 2)
            logging.info(
                f"{LEVEL_2}✅ before: {before_deletion} | after: {after_deletion} | diff: {diff} ({percent}%) => {dup_vars}"
            )

        else:
            logging.info(f"{LEVEL_2}{SKIP}")

        logging.info(f"{LEVEL_1}🔹 Remove redundant variables")

        if params.redundant_variables is not None:
            before_deletion = len(dataset.columns)
            dataset.drop(columns=params.redundant_variables, inplace=True)
            after_deletion = len(dataset.columns)
            diff = before_deletion - after_deletion
            percent = round((diff / before_deletion) * 100, 2)
            logging.info(
                f"{LEVEL_2}✅ before: {before_deletion} | after: {after_deletion} | diff: {diff} ({percent}%) => {params.redundant_variables}"
            )
        else:
            logging.info(f"{LEVEL_2}{SKIP}")

    else:
        logging.info(f"{LEVEL_1}{SKIP}")

    turnoff_verbose()


def remove_missing_values_variables(dataset: DataFrame, threshold: float):
    many_missing_data_variables_report = [
        (
            col_name,
            dataset[col_name].isna().sum() / len(dataset),
            (dataset[col_name].isna().sum() / len(dataset)) > threshold,
        )
        for col_name in dataset.columns
    ]
    many_missing_data_variables = [
        col_name
        for (col_name, _, will_removed) in many_missing_data_variables_report
        if will_removed
    ]
    report_dataframe = pd.DataFrame(many_missing_data_variables_report)
    report_dataframe.columns = [
        "Variable",
        "Missing Data Proportion",
        "Greater than th?",
    ]
    log_dataset(report_dataframe, LEVEL_3)
    dataset.drop(columns=many_missing_data_variables, inplace=True)


def remove_missing_values_records(dataset: DataFrame, threshold: float):
    missing_records_proportion = dataset.isnull().any(axis=1).sum() / len(dataset)
    if missing_records_proportion < threshold:
        logging.info(
            f"{LEVEL_3}📊 missing < th ({missing_records_proportion} < {threshold})"
        )
        dataset.dropna(inplace=True)
    else:
        logging.info(
            f"{LEVEL_3}📊 missing > th ({missing_records_proportion} > {threshold})"
        )
        dataset["sum_of_nulls"] = dataset.isnull().sum(axis=1)
        sorted_dataset = dataset.sort_values(by="sum_of_nulls", ascending=False)
        sorted_dataset_indexes = sorted_dataset.index
        indexes_to_be_removed = sorted_dataset_indexes[
            0 : floor(len(dataset) * threshold)
        ]
        dataset.drop(indexes_to_be_removed, axis=0, inplace=True)


def fill_with_value(
    dataset: DataFrame,
    fill_value: PRIMITIVES,  # type: ignore
    to_names: list[str] | None,
    to_types: list[VariableType] | None,
):
    if to_names is not None:
        for column in to_names:
            dataset[column].fillna(fill_value, inplace=True)

        logging.info(f"{LEVEL_3}✅ Fill with '{fill_value}' to {to_names}")

    else:
        logging.info(f"{LEVEL_3}{SKIP_TO_NAMES}")

    if to_types is not None:
        pandas_types = get_pandas_types(to_types)
        columns = list(dataset.select_dtypes(include=pandas_types).columns)

        for column in columns:
            dataset[column].fillna(fill_value, inplace=True)

        logging.info(
            f"{LEVEL_3}✅ Fill with '{fill_value}' to types {list(map(lambda x: x.value, to_types))} -> {columns}"
        )

    else:
        logging.info(f"{LEVEL_3}{SKIP_TO_TYPES}")


def handle_missing_data(
    dataset: DataFrame, transformation_verbose: bool, params: HandleMissingData | None
):
    setup_verbose(transformation_verbose)
    logging.info("(5) HANDLE MISSING DATA:")

    if params is not None:
        logging.info(f"{LEVEL_1}🔹 Remove")

        if params.remove is not None:
            logging.info(f"{LEVEL_2}🔹 Variables up to the threshold")

            if params.remove.variables_threshold is not None:
                before_deletion = len(dataset.columns)
                remove_missing_values_variables(
                    dataset, params.remove.variables_threshold
                )
                after_deletion = len(dataset.columns)
                diff = before_deletion - after_deletion
                percent = round((diff / before_deletion) * 100, 2)
                logging.info(
                    f"{LEVEL_3}✅ For th: {params.remove.variables_threshold} | before: {before_deletion} | after: {after_deletion} | diff: {diff} ({percent}%)"
                )

            else:
                logging.info(f"{LEVEL_3}{SKIP}")

            logging.info(f"{LEVEL_2}🔹 Records up to the threshold")

            if params.remove.records_threshold is not None:
                before_deletion = len(dataset)
                remove_missing_values_records(dataset, params.remove.records_threshold)
                after_deletion = len(dataset)
                diff = before_deletion - after_deletion
                percent = round((diff / before_deletion) * 100, 2)
                logging.info(
                    f"{LEVEL_3}✅ For th: {params.remove.variables_threshold} | before: {before_deletion} | after: {after_deletion} | diff: {diff} ({percent}%)"
                )
            else:
                logging.info(f"{LEVEL_3}{SKIP}")

        else:
            logging.info(f"{LEVEL_2}{SKIP}")

        logging.info(f"{LEVEL_1}🔹 Fill")

        if params.fill is not None:
            for index, fill in enumerate(params.fill):
                fill_method = fill.apply.fill_method
                fill_value = fill.apply.value
                to_names = fill.to.variable_names
                to_types = fill.to.variable_types

                logging.info(f"{LEVEL_2}🔹 Transformation #{index + 1}")

                if fill_method is not None:
                    apply_function(
                        dataset,
                        fill_method,
                        None,
                        to_names,
                        to_types,
                    )

                if fill_value is not None:
                    fill_with_value(dataset, fill_value, to_names, to_types)
        else:
            logging.info(f"{LEVEL_2}{SKIP}")

    else:
        logging.info(f"{LEVEL_1}{SKIP}")

    turnoff_verbose()


def get_outliers_by_z_score(dataset: DataFrame, column: str) -> list[list[int]]:
    ctype = dataset[column].dtype
    if ctype != np.int64 and ctype != np.float64:
        raise Exception(f"Column {column} is not of type: int64 or float64")
    mean = dataset[column].mean()
    std = np.std(dataset[column])
    threes = 3
    upper_outliers_ids = list[int](
        dataset[dataset[column].apply(lambda x: ((x - mean) / std) > threes)].index
    )
    lower_outliers_ids = list[int](
        dataset[
            dataset[column].apply(lambda x: ((x - mean) / std) < threes * (-1))
        ].index
    )

    outliers_ids = [lower_outliers_ids, upper_outliers_ids]

    return list[list[int]](outliers_ids)


def get_outliers_by_method(
    detect_method: NumericalDetectMethod,
    dataset: DataFrame,
    column: str,
) -> list[list[int]]:
    if detect_method == NumericalDetectMethod.Z_SCORE:
        return get_outliers_by_z_score(dataset, column)

    return get_outliers_by_z_score(dataset, column)  # replace by iqr


def get_outliers_by_limit(
    limit: float | int | type[datetime],
    is_upper: bool,
    dataset: DataFrame,
    column: str,
):  # Unused for now
    ctype = dataset[column].dtype
    if ctype != np.int64 and ctype != np.float64 and ctype != np.datetime64:
        raise Exception(f"Column {column} is not of type: int64, float64 or datetime64")

    outliers_ids = dataset[
        dataset[column].apply(lambda x: x > limit if is_upper else x < limit)
    ]

    return list[int](outliers_ids.index)


def fill_outliers_by_method(
    outliers_ids: list[int],
    apply: NumericalApplyMethod,
    is_lower: bool,
    column: str,
    dataset: DataFrame,
):
    if apply == NumericalApplyMethod.MEAN:
        mean_value = dataset[column].mean()
        dataset[column] = np.where(
            dataset.index.isin(outliers_ids), mean_value, dataset[column]
        )

    if apply == NumericalApplyMethod.QUANTILE:
        quantile_value = (
            dataset[column].quantile(0.1) if is_lower else dataset[column].quantile(0.9)
        )
        dataset[column] = np.where(
            dataset.index.isin(outliers_ids), quantile_value, dataset[column]
        )

    if apply == NumericalApplyMethod.REMOVE_RECORD:
        dataset = dataset[~dataset.index.isin(outliers_ids)]


def fill_outliers(
    outliers_ids: list[int],
    apply: NumericalApplyMethod | None,
    apply_limit: float | int | type[datetime] | None,
    is_lower: bool,
    column: str,
    dataset: DataFrame,
):
    if len(outliers_ids) > 0:
        if apply is not None:
            fill_outliers_by_method(outliers_ids, apply, is_lower, column, dataset)
            logging.info(
                f'{LEVEL_4}🔹 Detected {"LOWER" if is_lower else "UPPER"} outliers in "{column}" => ids {outliers_ids}'
            )
    else:
        logging.info(
            f'{LEVEL_4}🔹 NOT detected {"LOWER" if is_lower else "UPPER"} outliers in "{column}"'
        )


def apply_handle_outliers(
    detect_method: NumericalDetectMethod,
    apply_lower: NumericalApplyMethod | None,
    apply_upper: NumericalApplyMethod | None,
    apply_lower_limit: float | int | type[datetime] | None,
    apply_upper_limit: float | int | type[datetime] | None,
    to_names: list[str] | None,
    to_types: list[VariableType] | None,
    dataset: DataFrame,
):
    if to_names is not None:
        for column in to_names:
            pass

        # logging.info(f"{LEVEL_3}✅ Fill with '{fill_value}' to {to_names}")

    else:
        logging.info(f"{LEVEL_3}{SKIP_TO_NAMES}")

    if to_types is not None:
        pandas_types = get_pandas_types(to_types)
        columns = list(dataset.select_dtypes(include=pandas_types).columns)

        logging.info(
            f'{LEVEL_3}✅ Use "{detect_method}" in types {list(map(lambda x: x.value, to_types))} -> {columns}'
        )

        for column in columns:
            method_ids = get_outliers_by_method(detect_method, dataset, column)
            lower_methods_ids = method_ids[0]
            upper_methods_ids = method_ids[1]

            fill_outliers(
                lower_methods_ids,
                apply_lower,
                apply_lower_limit,
                True,
                column,
                dataset,
            )
            fill_outliers(
                upper_methods_ids,
                apply_upper,
                apply_upper_limit,
                False,
                column,
                dataset,
            )

    else:
        logging.info(f"{LEVEL_3}{SKIP_TO_TYPES}")


def handle_outliers(
    dataset: DataFrame, transformation_verbose: bool, params: HandleOutliers | None
):
    setup_verbose(transformation_verbose)
    logging.info("(6) HANDLE OUTLIERS:")

    if params is not None:
        logging.info(f"{LEVEL_1}🔹 Outliers for Numerical Variables")

        if params.numerical is not None:
            for index, outlier in enumerate(params.numerical):
                logging.info(f"{LEVEL_2}🔹 Transformation #{index + 1}")

                detect_method = outlier.detect.method
                apply_upper_limit = outlier.apply.upper_limit
                apply_lower_limit = outlier.apply.lower_limit

                apply_lower = outlier.apply.lower
                apply_upper = outlier.apply.upper

                to_names = outlier.to.variable_names
                to_types = outlier.to.variable_types

                apply_handle_outliers(
                    detect_method,
                    apply_lower,
                    apply_upper,
                    apply_lower_limit,
                    apply_upper_limit,
                    to_names,
                    to_types,
                    dataset,
                )

        else:
            logging.info(f"{LEVEL_2}{SKIP}")

    else:
        logging.info(f"{LEVEL_1}{SKIP}")

    turnoff_verbose()


def main():
    # Read config file ----------------------------------------------------
    dt_config = read_dt_config_file("data-cleaning.yml")
    handled_dt_config: DataCleaningConfigFile = handle_dt_config(dt_config)
    # pprint(handled_dt_config.get(), sort_dicts=False)

    # Read original dataset -----------------------------------------------
    kwargs = {"params": handled_dt_config.general.input}
    original_dataset = read_dataset(**kwargs)
    # logging.info(original_dataset)

    # 1. PREPARE VARIABLE NAMES
    t1 = test(
        params={
            "params": handled_dt_config.stages.prepare.variable_names
            if handled_dt_config.stages.prepare is not None
            else None
        },
        transformation=prepare_var_names,
        original_dataset=original_dataset,
        d_is_returned=True,
        transformation_verbose=True,
    )

    # 2. PREPARE VARIABLE VALUES
    t2 = test(
        params={
            "params": handled_dt_config.stages.prepare.variable_values
            if handled_dt_config.stages.prepare is not None
            else None
        },
        transformation=prepare_var_values,
        original_dataset=t1,
        d_is_returned=True,
        transformation_verbose=True,
    )

    # 3. HANDLE DUPLICATE DATA
    t3 = test(
        params={"params": handled_dt_config.stages.remove_duplicate},
        transformation=handle_duplicate_data,
        original_dataset=t2,
        d_is_returned=True,
        transformation_verbose=True,
    )

    # 4. HANDLE IRRELEVANT DATA
    t4 = test(
        params={"params": handled_dt_config.stages.remove_irrelevant},
        transformation=handle_irrelevant_data,
        original_dataset=t3,
        d_is_returned=True,
        transformation_verbose=True,
    )

    # 4. HANDLE IRRELEVANT DATA
    t5 = test(
        params={"params": handled_dt_config.stages.handle_missing_data},
        transformation=handle_missing_data,
        original_dataset=t4,
        transformation_verbose=True,
        d_is_returned=True,
    )

    # 4. HANDLE IRRELEVANT DATA
    test(
        params={"params": handled_dt_config.stages.handle_outliers},
        transformation=handle_outliers,
        original_dataset=t5,
        transformation_verbose=True,
    )


if __name__ == "__main__":
    main()
