import marimo as mo
import pandas as pd
import altair as alt
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Union, Tuple
from scipy.stats import norm as normf
from sklearn import mixture
# from oss_app.dataset import Dataset
from scipy.stats import ks_2samp
from sklearn import decomposition
from matplotlib.lines import Line2D

# %% Helper functions for oss_app.py notebook


def mo_print(*args, **kwargs):
    """
    Prints messages to a marimo output cell using mo.redirect_stdout().
    Accepts the same arguments as the built-in print() function.
    """
    with mo.redirect_stdout():
        print(*args, **kwargs)


def today_date(time_included: bool = False) -> str:
    """
    Returns today's date in the format YYYY-MM-DD.
    """
    if time_included:
        return datetime.now().strftime("%y%m%d_%Hh%Mm")
    return datetime.now().strftime("%Y%m%d")


def save_parameters(
    params_output: dict, location: str | Path = None, date_string: str = None, overwrite: bool = False
) -> Path:
    if location is None:
        location = Path("oss_app/params")
    elif not isinstance(location, Path):
        location = Path(location)

    if not location.exists():
        location.mkdir(parents=True, exist_ok=True)
        mo_print(f"Created directory: {location}")

    if not date_string:
        date_string = today_date(time_included=False)

    file_out_name = f"params_{date_string}.json"
    file_out_path = location / file_out_name
    if not overwrite:
        i = 0
        while file_out_path.exists():  # if file already exist, append numbers
            file_out_path = location / f"params_{date_string}_{i}.json"
            i += 1

    mo_print(f"Saving parameters to `{file_out_path}`...")
    with open(file_out_path, "w") as file:
        json.dump(params_output, file, indent=4)
    mo_print("...file saved.")
    return file_out_path


# %% Data processing helper functions


def fix_name(*names: str) -> Union[str, Tuple[str, ...]]:
    """
    Fixes one or more names by removing leading/trailing whitespace and converting to lowercase.
    If one name is passed, returns a string.
    If multiple names are passed, returns a tuple of strings.
    """
    fixed_names = [name.strip().lower() for name in names]
    if len(fixed_names) == 1:
        return fixed_names[0]
    return tuple(fixed_names)


def fix_column_names(df: pd.DataFrame, *additional_df: Tuple[pd.DataFrame, ...]):
    """
    Fix column names by removing leading/trailing whitespace and converting to lowercase.
    """
    df.columns = [fix_name(col) for col in df.columns]
    for additional in additional_df:
        additional.columns = [fix_name(col) for col in additional.columns]


def make_categorical(series: pd.Series) -> dict:
    """
    Creates a dictionary with codes and labels for a categorical series.
    Leveraged in plotting functions

    Args:
        series: The pandas Series to convert.

    Returns:
        A dictionary containing the categorical codes and labels (categories).
    """
    categorical_series = pd.Categorical(series)
    return {"codes": categorical_series.codes, "labels": categorical_series.categories}


# TODO: Logging helper functions
