from datetime import datetime
from enum import (
    Enum,
)

# Typing to test config file
from typing import (
    Any,
    Callable,
    Type,
    TypeVar,
    Union,
)

# utils

U = TypeVar("U")

# Types

DATETIME_TYPE = type(datetime)

PRIMITIVES = Union[bool, str, int, float, type(None), datetime]


def is_primitive(obj: Any) -> bool:
    return isinstance(obj, PRIMITIVES)


def optional(obj: Any, key: str, cls: Callable[[Any], U], cls_name: str) -> U | None:
    if key in obj:
        if obj[key] is None:
            return None

        if not isinstance(obj[key], list):
            return cls(obj[key])

        raise Exception(
            f'ERROR: optional field "{key}" must not be a list in "{cls_name}"'
        )
    return None


def optional_list(
    obj: Any, key: str, cls: Callable[[Any], U], cls_name: str
) -> list[U] | None:
    if key in obj:
        if obj[key] is None:
            return None
        if isinstance(obj[key], list):
            return list(map(lambda item: cls(item), obj[key]))
        raise Exception(f'ERROR: optional field "{key}" must be a list in "{cls_name}"')
    return None


def required(obj: Any, key: str, cls: Callable[[Any], U], cls_name: str) -> U:
    if key in obj:
        if obj[key] is not None and not isinstance(obj[key], list):
            return cls(obj[key])
        raise Exception(
            f'ERROR: required field "{key}" must not be a list in "{cls_name}"'
        )
    raise Exception(f'ERROR: missing required field "{key}" in "{cls_name}"')


def required_list(
    obj: Any, key: str, cls: Callable[[Any], U], cls_name: str
) -> list[U]:
    if key in obj:
        if obj[key] is not None and isinstance(obj[key], list):
            return list(map(lambda item: cls(item), obj[key]))
        raise Exception(f'ERROR: required field "{key}" must be a list in "{cls_name}"')
    raise Exception(f'ERROR: missing required field "{key}" in "{cls_name}"')


# Get values for printing


def get_optional(param: list[Any] | Any | None):
    if isinstance(param, list):
        return list(map(lambda i: i.get() if not is_primitive(i) else i, param))
    if is_primitive(param):
        return param
    return param.get() if param is not None else None


def value_optional(param: list[Any] | Any | None):
    if isinstance(param, list):
        return list(map(lambda i: i.value, param))
    return param.value if param is not None else None


def identity_function(param: Any) -> Any:
    return param


def test_at_least_one_not_none_field(self: Any, cls_name: str):
    at_least_one_not_none: bool = any(
        value is not None for (_, value) in self.__dict__.items()
    )
    if not at_least_one_not_none:
        raise Exception(f'ERROR: At least one field in "{cls_name}" must not be None')


# Enums


class VariableNamesApply(str, Enum):
    LOWER = "lower"
    UPPER = "upper"


class VariableValuesApplyFunction(str, Enum):
    INT = "int"
    STR = "str"
    FLOAT = "float"
    BOOL = "bool"
    ROUND = "round"
    UPPER = "upper"
    LOWER = "lower"
    DATETIME = "datetime"


class VariableType(str, Enum):
    FLOAT = "float"
    INT = "int"
    BOOL = "bool"
    DATETIME = "datetime"
    STR = "str"


class FillMethod(str, Enum):
    BFILL = "bfill"
    FFILL = "ffill"
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    INTERPOLATE = "interpolate"
    REMOVE_R = "remove-r"


class NumericalDetectMethod(str, Enum):
    Z_SCORE = "z-score"
    IQR = "iqr"


class NumericalApplyMethod(str, Enum):
    REMOVE_RECORD = "remove-r"
    QUANTILE = "quantile"
    MEAN = "mean"


# Classes


class VariableValuesApplyParams:
    def __init__(self, params: Any):
        self.digits_to_round = optional(
            cls=int, key="digits_to_round", obj=params, cls_name="params"
        )

    def get(self):
        return {
            "digits_to_round": self.digits_to_round,
        }

    digits_to_round: int | None


def check_required_params(
    functions: list[VariableValuesApplyFunction],
    params: VariableValuesApplyParams | None,
):
    is_round_in_functions = VariableValuesApplyFunction.ROUND in functions

    if params is not None:
        if is_round_in_functions and params.digits_to_round is None:
            raise Exception(
                "digits_to_round param must not be None when round function is selected"
            )

    if params is None and is_round_in_functions:
        raise Exception("params must be specified")


class To:
    def __init__(self, to: Any):
        self.variable_names = optional_list(
            cls=str, key="variable_names", obj=to, cls_name="to"
        )
        self.variable_types = optional_list(
            cls=VariableType, key="variable_types", obj=to, cls_name="to"
        )
        test_at_least_one_not_none_field(self, cls_name="to")

    def get(self):
        return {
            "variable_names": get_optional(self.variable_names),
            "variable_types": get_optional(self.variable_types),
        }

    variable_names: list[str] | None
    variable_types: list[VariableType] | None


class VariableValuesApply:
    def __init__(self, variable_values: Any):
        self.function = required_list(
            cls=VariableValuesApplyFunction,
            cls_name="variable_values",
            key="function",
            obj=variable_values,
        )
        temp_params = optional(
            cls=VariableValuesApplyParams,
            key="params",
            obj=variable_values,
            cls_name="variable_values",
        )

        check_required_params(self.function, temp_params)

        self.params = temp_params

    def get(self):
        return {
            "function": value_optional(self.function),
            "params": get_optional(self.params),
        }

    function: list[VariableValuesApplyFunction]
    params: VariableValuesApplyParams | None


class VariableNames:
    def __init__(self, variable_names: Any):
        self.apply = required(
            cls=VariableNamesApply,
            cls_name="variable_names",
            key="apply",
            obj=variable_names,
        )

    def get(self):
        return {
            "apply": self.apply.value,
        }

    apply: VariableNamesApply


class VariableValues:
    def __init__(self, variable_values: Any):
        self.apply = required(
            cls=VariableValuesApply,
            cls_name="variable_values",
            key="apply",
            obj=variable_values,
        )
        self.to = required(
            cls=To, cls_name="variable_values", key="to", obj=variable_values
        )

    def get(self):
        return {
            "apply": self.apply.get(),
            "to": self.to.get(),
        }

    apply: VariableValuesApply
    to: To


class Prepare:
    def __init__(self, prepare: Any):
        self.variable_names = optional(
            cls=VariableNames, key="variable_names", obj=prepare, cls_name="prepare"
        )
        self.variable_values = optional_list(
            cls=VariableValues, key="variable_values", obj=prepare, cls_name="prepare"
        )

    def get(self):
        return {
            "variable_names": get_optional(self.variable_names),
            "variable_values": get_optional(self.variable_values),
        }

    variable_names: VariableNames | None
    variable_values: list[VariableValues] | None


class RemoveDuplicate:
    def __init__(self, remove_duplicate: Any):
        self.records = optional(
            cls=bool, key="records", obj=remove_duplicate, cls_name="remove_duplicate"
        )
        self.variables = optional(
            cls=bool, key="variables", obj=remove_duplicate, cls_name="remove_duplicate"
        )

    def get(self):
        return {
            "records": self.records,
            "variables": self.variables,
        }

    records: bool | None
    variables: bool | None


class RemoveIrrelevant:
    def __init__(self, remove_irrelevant: Any):
        self.one_level_variables = optional(
            cls=bool,
            key="one_level_variables",
            obj=remove_irrelevant,
            cls_name="remove_irrelevant",
        )
        self.redundant_variables = optional_list(
            cls=str,
            key="redundant_variables",
            obj=remove_irrelevant,
            cls_name="remove_irrelevant",
        )

    def get(self):
        return {
            "one_level_variables": self.one_level_variables,
            "redundant_variables": get_optional(self.redundant_variables),
        }

    one_level_variables: bool | None
    redundant_variables: list[str] | None


class Remove:
    def __init__(self, remove: Any):
        self.records_threshold = optional(
            cls=float, key="records_threshold", obj=remove, cls_name="remove"
        )
        self.variables_threshold = optional(
            cls=float, key="variables_threshold", obj=remove, cls_name="remove"
        )

    def get(self):
        return {
            "records_threshold": self.records_threshold,
            "variables_threshold": self.variables_threshold,
        }

    records_threshold: float | None
    variables_threshold: float | None


class FillApply:
    def __init__(self, apply: Any):
        self.fill_method = optional_list(
            cls=FillMethod, key="fill_method", obj=apply, cls_name="apply"
        )
        self.value = optional(
            cls=identity_function, key="value", obj=apply, cls_name="apply"
        )
        test_at_least_one_not_none_field(self, cls_name="apply")

    def get(self):
        return {
            "fill_method": list(map(lambda i: i.value, self.fill_method))
            if self.fill_method is not None
            else None,
            "value": self.value,
        }

    fill_method: list[FillMethod] | None
    value: float | int | bool | Type[datetime] | str | None


class Fill:
    def __init__(self, fill: Any):
        self.apply = required(cls=FillApply, cls_name="fill", key="apply", obj=fill)
        self.to = required(cls=To, cls_name="fill", key="to", obj=fill)

    def get(self):
        return {
            "apply": self.apply.get(),
            "to": self.to.get(),
        }

    apply: FillApply
    to: To


class HandleMissingData:
    def __init__(self, handle_missing_data: Any):
        self.remove = optional(
            cls=Remove,
            key="remove",
            obj=handle_missing_data,
            cls_name="handle_missing_data",
        )
        self.fill = optional_list(
            cls=Fill,
            key="fill",
            obj=handle_missing_data,
            cls_name="handle_missing_data",
        )

    def get(self):
        return {
            "remove": get_optional(self.remove),
            "fill": get_optional(self.fill),
        }

    remove: Remove | None
    fill: list[Fill] | None


class NumericalDetect:
    def __init__(self, detect: Any):
        self.method = required(
            cls=identity_function, key="method", obj=detect, cls_name="detect"
        )

    def get(self):
        return {
            "method": self.method,
        }

    method: NumericalDetectMethod


class NumericalApply:
    def __init__(self, apply: Any):
        self.lower = optional(
            cls=NumericalApplyMethod, key="lower", obj=apply, cls_name="apply"
        )
        self.upper = optional(
            cls=NumericalApplyMethod, key="upper", obj=apply, cls_name="apply"
        )
        self.lower_limit = optional(
            cls=identity_function, key="lower_limit", obj=apply, cls_name="detect"
        )
        self.upper_limit = optional(
            cls=identity_function, key="upper_limit", obj=apply, cls_name="detect"
        )

    def get(self):
        return {
            "lower": value_optional(self.lower),
            "upper": value_optional(self.upper),
            "lower_limit": self.lower_limit,
            "upper_limit": self.upper_limit,
        }

    lower: NumericalApplyMethod | None
    upper: NumericalApplyMethod | None
    lower_limit: float | int | Type[datetime] | None
    upper_limit: float | int | Type[datetime] | None


class Numerical:
    def __init__(self, numerical: Any):
        self.detect = required(
            cls=NumericalDetect, cls_name="numerical", key="detect", obj=numerical
        )
        self.apply = required(
            cls=NumericalApply, cls_name="numerical", key="apply", obj=numerical
        )
        self.to = required(cls=To, cls_name="numerical", key="to", obj=numerical)

    def get(self):
        return {
            "detect": self.detect.get(),
            "apply": self.apply.get(),
            "to": self.to.get(),
        }

    detect: NumericalDetect
    apply: NumericalApply
    to: To


class HandleOutliers:
    def __init__(self, handle_outliers: Any):
        self.numerical = required_list(
            cls=Numerical,
            cls_name="handle_outliers",
            key="numerical",
            obj=handle_outliers,
        )

    def get(self):
        return {
            "numerical": list(map(lambda i: i.get(), self.numerical)),
        }

    numerical: list[Numerical]


class FixTyposApply:
    def __init__(self, apply: Any):
        self.fixed_value = required(
            cls=str, cls_name="apply", key="fixed_value", obj=apply
        )
        self.incorrect_values = required_list(
            cls=str,
            cls_name="apply",
            key="incorrect_values",
            obj=apply,
        )

    def get(self):
        return {
            "fixed_value": self.fixed_value,
            "incorrect_values": list(map(lambda i: i, self.incorrect_values)),
        }

    fixed_value: str
    incorrect_values: list[str]


class FixTypos:
    def __init__(self, fix_typos: Any):
        self.apply = required_list(
            cls=FixTyposApply,
            cls_name="fix_typos",
            key="apply",
            obj=fix_typos,
        )
        self.to = required(cls=To, cls_name="fix_typos", key="to", obj=fix_typos)

    def get(self):
        return {
            "apply": list(map(lambda i: i.get(), self.apply)),
            "to": self.to.get(),
        }

    apply: list[FixTyposApply]
    to: To


class General:
    def __init__(self, general: Any):
        self.input = required(cls=str, cls_name="general", key="input", obj=general)
        self.output = required(cls=str, cls_name="general", key="output", obj=general)

    def get(self):
        return {
            "input": self.input,
            "output": self.output,
        }

    input: str
    output: str


class Stages:
    def __init__(self, stages: Any):
        self.prepare = optional(
            cls=Prepare, key="prepare", obj=stages, cls_name="stages"
        )
        self.remove_duplicate = optional(
            cls=RemoveDuplicate, key="remove_duplicate", obj=stages, cls_name="stages"
        )
        self.remove_irrelevant = optional(
            cls=RemoveIrrelevant, key="remove_irrelevant", obj=stages, cls_name="stages"
        )
        self.handle_missing_data = optional(
            cls=HandleMissingData,
            key="handle_missing_data",
            obj=stages,
            cls_name="stages",
        )
        self.handle_outliers = optional(
            cls=HandleOutliers, key="handle_outliers", obj=stages, cls_name="stages"
        )
        # self.fix_typos = optional_list(
        #    cls=FixTypos, key="fix_typos", obj=stages, cls_name="stages"
        # )
        # test_at_least_one_not_none_field(self, cls_name="stages")

    def get(self):
        return {
            "prepare": get_optional(self.prepare),
            "remove_duplicate": get_optional(self.remove_duplicate),
            "remove_irrelevant": get_optional(self.remove_irrelevant),
            "handle_missing_data": get_optional(self.handle_missing_data),
            "handle_outliers": get_optional(self.handle_outliers),
            # "fix_typos": get_optional(self.fix_typos),
        }

    prepare: Prepare | None
    remove_duplicate: RemoveDuplicate | None
    remove_irrelevant: RemoveIrrelevant | None
    handle_missing_data: HandleMissingData | None
    handle_outliers: HandleOutliers | None
    # fix_typos: list[FixTypos] | None # (NOT PLANED YET)


class DataCleaningConfigFile:
    def __init__(self, data_cleaning_config_file: Any):
        self.general = required(
            cls=General,
            cls_name="config_file",
            key="general",
            obj=data_cleaning_config_file,
        )
        self.stages = required(
            cls=Stages,
            cls_name="config_file",
            key="stages",
            obj=data_cleaning_config_file,
        )

    def get(self):
        return {
            "general": self.general.get(),
            "stages": self.stages.get(),
        }

    general: General
    stages: Stages
