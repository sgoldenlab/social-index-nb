import marimo as mo
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Union, Tuple


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


def test_minmax(plotdata: np.ndarray, curr_mm=(0, 0), debug=False):
    curr_min, curr_max = curr_mm
    vmin, vmax = (min(plotdata), max(plotdata))
    mintest = 1 if vmin < curr_min else 0
    maxtest = 1 if vmax > curr_max else 0
    match (mintest, maxtest):
        case (1, 0):
            curr_min = vmin
        case (0, 1):
            curr_max = vmax
        case (1, 1):
            curr_min = vmin
            curr_max = vmax
        case _:
            curr_min, curr_max = vmin, vmax
    if debug:
        print(mintest, maxtest)
    return curr_min, curr_max


def round_to_nearest(base, *nums, round_up=True):
    """round num to nearest `base`,
    round up will round number away to higher magnitude, away from 0.
    """
    for num in nums:
        if (num < 0 and round_up) or (num > 0 and not round_up):
            # rounds negative number up, positive down
            bias = base//2
        else:
            # rounds negative number down, positive up
            bias = base
        roundt = num % base
        n = num + bias if roundt != 0 else num
        yield n - (n % base)


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
