# Data Cleaning Library Documentation

## Overview

The data cleaning library provides a convenient way to automate the data cleaning process. It uses a YAML configuration file to specify various cleaning stages and parameters. Below is an example of how to use the library and an explanation of the YAML configuration file.

### Example Usage

```python
import os
from ds_toolkit.data_cleaning import clean

root_dir = os.path.dirname(os.path.abspath(__file__))
relative_config_path = "data_cleaning.yml"
intermediate_steps = clean(
    root_dir,
    relative_config_path,
)
```

## YAML Configuration File (data_cleaning.yml)

```yml
general:
  input: 'test_data.csv'
  output: 'cleaned_dataset.csv'

stages:
  prepare:
    variable_names:
      apply: 'lower'
    variable_values:
      - apply:
          function: ['lower']
          params:
            digits_to_round: null
        to:
          variable_types: ['str']
          variable_names: null

  remove_duplicate:
    records: true
    variables: true

  remove_irrelevant:
    one_level_variables: true
    redundant_variables: ['month']

  handle_missing_data:
    remove:
      variables_threshold: 0.3
      records_threshold: 0.3
    fill:
      - apply:
          fill_method: ['mean']
          value: null
        to:
          variable_types: ['float', 'int']
          variable_names: null

  handle_outliers:
    numerical:
      - detect:
          method: 'z-score'
        apply:
          lower: 'quantile'
          upper: 'quantile'
          lower_limit: null
          upper_limit: null
        to:
          variable_types: ['float', 'int']
          variable_names: null
```
## YAML Configuration Explanation

### General Section

- **input**: Path to the input dataset (CSV format).
- **output**: Path to save the cleaned dataset (CSV format).

### Stages Section

#### 1. Prepare (Optional)

- **variable_names** (Optional): Apply case conversion to variable names.
  - **apply**: Specify the case conversion ('upper' or 'lower').

- **variable_values** (Optional): Apply transformations to variable values.
  - **apply**: Specify the transformation function(s) ('int', 'str', 'float', 'bool', 'round', 'upper', 'lower', 'datetime').
  - **params**: Parameters for the transformation function(s).

#### 2. Remove Duplicate (Optional)

- **records**: Remove duplicate records if set to `true`.
- **variables**: Remove duplicate variables if set to `true`.

#### 3. Remove Irrelevant (Optional)

- **one_level_variables**: Remove one-level variables if set to `true`.
- **redundant_variables**: Specify a list of redundant variable names to be removed.

#### 4. Handle Missing Data (Optional)

- **remove**: Remove variables or records with missing data based on specified thresholds.
  - **variables_threshold**: Threshold for removing variables with missing data.
  - **records_threshold**: Threshold for removing records with missing data.

- **fill**: Fill missing data using various methods.
  - **apply**: Specify the fill method(s) ('bfill', 'ffill', 'mean', 'median', 'mode', 'interpolate', 'remove-r').
  - **value**: Value to fill missing data with (if applicable).
  - **to**: Specify the variable types and names to apply the fill method.

#### 5. Handle Outliers (Optional)

- **numerical**: Handle outliers for numerical variables.
  - **detect**: Specify the detection method ('z-score', 'iqr').
  - **apply**: Specify the action for outliers ('remove-r', 'quantile', 'mean').
  - **lower**: Specify the lower bound action.
  - **upper**: Specify the upper bound action.
  - **lower_limit**: Specify the lower limit for outliers.
  - **upper_limit**: Specify the upper limit for outliers.
  - **to**: Specify the variable types and names to apply outlier handling.

Adjust the parameters and stages based on specific data cleaning requirements.

---

This documentation provides a comprehensive guide on using the data cleaning library and customizing the cleaning process according to specific requirements.
