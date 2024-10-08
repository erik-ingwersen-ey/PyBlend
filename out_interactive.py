from __future__ import annotations

import re
from typing import List, Dict, Any, Iterable
import subprocess
import logging
from pathlib import Path
import json

import numpy as np
import pandas as pd

import xlwings as xw
from matplotlib import pyplot as plt


def check_is_file(*filepaths):
    """Check if the provided file paths exist as files.

    This function takes one or more file paths as input and checks whether
    each path corresponds to an existing file. If any of the specified paths
    do not point to a valid file, a FileNotFoundError is raised, detailing
    which files were not found.

    Args:
        *filepaths (str): One or more file paths to check.

    Raises:
        FileNotFoundError: If any of the provided file paths do not correspond
    """

    files_not_found = []
    for filepath in filepaths:
        _filepath = Path(filepath)
        if not _filepath.is_file():
            files_not_found.append(filepath)
    if len(files_not_found) > 0:
        plural = "" if len(files_not_found) == 1 else "s"
        past_tense_form = "was" if len(files_not_found) == 1 else "were"
        exception_message = (
            f"The following file{plural} {past_tense_form}n't found: "
            + ", ".join(files_not_found)
        )
        raise FileNotFoundError(exception_message)


def is_integer(value):
    """Check if the given value is an integer.

    This function determines whether the provided value is an integer. It
    handles various types of inputs, including integers, strings that
    represent numeric values, floats, and iterables. For strings, it checks
    if all characters are numeric or if they match a specific pattern that
    indicates they represent an integer. For floats, it checks if the float
    is equivalent to an integer. If the input is an iterable, it recursively
    checks each element to see if they are integers.

    Parameters
    ----------
    value : Union[int, str, float, Iterable]
        The value to be checked for integer status.

    Returns
    -------
    bool
        True if the value is an integer or can be interpreted as an integer;
        otherwise, False.
    """

    if isinstance(value, int):
        return True
    if isinstance(value, str):
        if all(c.isnumeric() for c in value) or re.match(r"\d+\.0+[^1-9]$", value):
            return True
        return False

    if isinstance(value, float):
        return int(value) == value

    if isinstance(value, Iterable):
        return all(is_integer(v) for v in value)
    return False


def autofit_columns_from_sheet(sheet, min_width=10):
    """Autofit all columns in an Excel sheet.

    This function adjusts the width of all columns in the specified Excel
    sheet to fit the content. It ensures that each column is at least a
    specified minimum width. The autofitting process is performed by
    checking the width of each column and adjusting it accordingly.

    Parameters
    ----------
    sheet : xlwings.Sheet
        The sheet where columns need to be autofit.
    min_width : int, default=10
        The minimum width for any column.
    """
    # Get the last used column in the sheet
    last_used_column = sheet.used_range.last_cell.column

    # Loop through each column from the first to the last used column
    for col in range(1, last_used_column + 1):
        column_range = sheet.range(f"{xw.utils.col_name(col)}:{xw.utils.col_name(col)}")
        column_range.autofit()

        # Check if the autofit width is less than the minimum width
        if column_range.column_width < min_width:
            column_range.column_width = min_width


def format_excel_sheet(sheet: xw.Sheet):
    """Format all used cells in an Excel sheet.

    This function applies specific styles to the header row, alternating row
    colors, and a distinct style for the last row of the provided Excel
    sheet. It clears any existing formats before applying the new styles,
    ensuring a consistent appearance throughout the sheet. The header row is
    styled with a blue background and white bold text, while the other rows
    have alternating colors for better readability. The last row is
    highlighted with a different color and bold text to distinguish it from
    the other rows.

    Parameters
    ----------
    sheet : xlwings.Sheet
        The sheet to be formatted.
    """
    sheet.used_range.clear_formats()

    # Get last used row and column
    first_column = sheet.used_range.columns[0].column
    first_row = sheet.used_range.columns[0].row
    last_column = sheet.used_range.last_cell.column
    last_row = sheet.used_range.last_cell.row
    first_column_name = xw.utils.col_name(first_column)
    last_column_name = xw.utils.col_name(last_column)

    # Formatting for the header row
    header_range = sheet.range(f"{first_column_name}{first_row}", f"{last_column_name}{first_row}")

    header_range.color = "#5B80B8"  # Fill color
    header_range.api.Font.Color = 0xFFFFFF  # Font color (White)
    header_range.api.Font.Bold = True
    header_range.api.Font.Size = 11
    header_range.api.Font.Name = "Calibri"
    header_range.api.HorizontalAlignment = xw.constants.HAlign.xlHAlignCenter
    header_range.api.Borders(xw.constants.BordersIndex.xlEdgeBottom).LineStyle = xw.constants.LineStyle.xlContinuous
    header_range.api.Borders(xw.constants.BordersIndex.xlEdgeBottom).Color = 0xFFFFFF  # White
    header_range.api.Borders(xw.constants.BordersIndex.xlEdgeBottom).Weight = xw.constants.BorderWeight.xlThick

    # Formatting for all row cells
    for row in range(2, last_row):  # Start from 2 to avoid header row
        row_range = sheet.range(f"{first_column_name}{row}", f"{last_column_name}{row}")
        row_range.api.Font.Size = 11
        row_range.api.Font.Name = "Calibri"
        row_range.api.Font.Color = 0x000000  # Black
        row_range.color = "#B9C8DE" if row % 2 == 0 else "#DEE6F0"
        row_range.api.HorizontalAlignment = xw.constants.HAlign.xlHAlignCenter
        row_range.api.Borders(xw.constants.BordersIndex.xlEdgeBottom).LineStyle = xw.constants.LineStyle.xlContinuous
        row_range.api.Borders(xw.constants.BordersIndex.xlEdgeBottom).Color = 0xFFFFFF  # White
        row_range.api.Borders(xw.constants.BordersIndex.xlEdgeBottom).Weight = xw.constants.BorderWeight.xlThin

    # Formatting for the last row
    last_row_range = sheet.range(f"{first_column_name}{last_row}", f"{last_column_name}{last_row}")
    last_row_range.color = "#4F81BD"
    last_row_range.api.Font.Color = 0xFFFFFF  # White
    last_row_range.api.Font.Bold = True
    last_row_range.api.Font.Size = 11
    last_row_range.api.Font.Name = "Calibri"
    last_row_range.api.HorizontalAlignment = xw.constants.HAlign.xlHAlignCenter
    last_row_range.api.Borders(xw.constants.BordersIndex.xlEdgeTop).LineStyle = xw.constants.LineStyle.xlDouble
    last_row_range.api.Borders(xw.constants.BordersIndex.xlEdgeTop).Color = 0xFFFFFF  # White
    last_row_range.api.Borders(xw.constants.BordersIndex.xlEdgeTop).Weight = xw.constants.BorderWeight.xlThick


def format_integer_columns(sheet: xw.Sheet):
    """Format columns in an Excel sheet where all values are integers.

    This function iterates through each column in the specified Excel sheet
    and checks if all values in the column are integers. If a column
    contains only integer values (or is empty), it applies a number format
    that includes a thousand separator and zero decimal places. This
    formatting helps improve the readability of integer data in the
    spreadsheet.

    Parameters
    ----------
    sheet xlwings.Sheet
        The sheet to be formatted.
    """
    # Get the last used column and row in the sheet
    last_column = sheet.used_range.last_cell.column
    last_row = sheet.used_range.last_cell.row

    # Iterate through each column in the used range
    for col_index in range(1, last_column + 1):
        column_range = sheet.range((2, col_index), (last_row, col_index))
        values = column_range.value

        # Check if all values in the column are integers
        if all(is_integer(val) or val is None or val == "" for val in values):
            # Apply a number format with thousands' separator and zero decimal places
            column_range.number_format = "#,##0"


# Convert input and output JSON files to dataframes
def explode_quality_rows(
    df: pd.DataFrame,
    quality_col_prefix: str = "quality_",
) -> pd.DataFrame:
    """Explode the 'quality_*' columns into rows.

    This function takes a DataFrame containing quality columns that hold
    dictionaries of quality parameters and transforms these columns into
    separate rows. Each dictionary in the quality columns is expanded into
    its own set of columns, while the remaining columns are repeated for
    each exploded row. This is useful for normalizing data where quality
    parameters are stored in a nested format.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe with the output specifications containing the quality
        columns that hold dictionaries of quality parameters to be extracted.
    quality_col_prefix : str, default='quality_'
        The prefix of the quality columns
        that contain the output pile specifications that need to be extracted
        into different columns.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with the quality dictionaries extracted
        into new columns, where each dictionary is represented as a separate
        row alongside the repeated original columns.
    """
    # Create a new DataFrame to store exploded rows
    exploded_df = pd.DataFrame()

    for index, row in df.iterrows():
        quality_dict_list = [
            value for key, value in row.items() if str(key).startswith(quality_col_prefix)
        ]

        # Convert the list of quality dictionaries into a DataFrame
        quality_df = pd.DataFrame(quality_dict_list)

        # Repeat the original columns for each exploded row
        repeated_columns = pd.DataFrame(
            [
                row.drop(
                    labels=[
                        col for col in df.columns if col.startswith(quality_col_prefix)
                    ]
                )
            ]
            * len(quality_df)
        )

        # Concatenate the repeated columns with the quality columns
        exploded_row_df = pd.concat(
            [
                repeated_columns.reset_index(drop=True),
                quality_df.reset_index(drop=True),
            ],
            axis=1,
        )

        # Append to the exploded DataFrame
        exploded_df = pd.concat([exploded_df, exploded_row_df], ignore_index=True)

    return exploded_df


def assign_engines_to_stockpiles(
    stockpiles_df: pd.DataFrame, engines_df: pd.DataFrame
) -> pd.DataFrame:
    """Assign engines to stockpiles based on matching yards and rails.

    This function takes two pandas DataFrames: one containing stockpile
    information and the other containing engine information. It iterates
    through each stockpile and assigns engines that match the specified yard
    and rail criteria. The result is an updated stockpile DataFrame that
    includes a new column listing the IDs of the assigned engines for each
    stockpile.

    Args:
        stockpiles_df (pd.DataFrame): A `pandas.DataFrame` containing stockpile
            information including 'rails' and 'yard'.
        engines_df (pd.DataFrame): A `pandas.DataFrame` containing engine information
            including 'yards' and 'rail'.

    Returns:
        pd.DataFrame: Updated `stockpiles_df` with an 'engines' column listing the
            assigned engine IDs.
    """
    stockpiles_df["engines"] = [[] for _ in range(stockpiles_df.shape[0])]

    for idx, stockpile in stockpiles_df.iterrows():
        assigned_engines = [
            eng_row["id"]
            for _, eng_row in engines_df.iterrows()
            if stockpile["yard"] in eng_row["yards"]
            and eng_row["rail"] in stockpile["rails"]
        ]
        stockpiles_df.at[idx, "engines"] = assigned_engines

    return stockpiles_df


def extract_quality_ini_values(
    stockpiles_df: pd.DataFrame, quality_prefix: str = "qualityIni"
) -> pd.DataFrame:
    """Extract initial quality values from nested dictionaries in a stockpiles
    DataFrame.

    This function processes a given DataFrame containing stockpile
    information, specifically focusing on columns that start with a
    specified prefix related to quality parameters. It extracts the
    'parameter' and 'value' from these nested dictionaries and adds them as
    new columns to the original DataFrame. The original quality columns are
    then removed from the DataFrame, resulting in a cleaner structure with
    individual quality parameters as separate columns.

    Args:
        stockpiles_df (pd.DataFrame): A `pandas.DataFrame` containing stockpile information,
            including quality parameters stored in nested dictionaries.
        quality_prefix (str?): Prefix used in column names for quality-related
            information. Defaults to 'qualityIni'.

    Returns:
        pd.DataFrame: Updated `stockpiles_df` with quality parameters extracted as individual
            columns.
    """
    quality_cols = stockpiles_df.columns[
        stockpiles_df.columns.str.startswith(quality_prefix)
    ]
    quality_ini_dict = {}

    for column in quality_cols:
        for idx, row in stockpiles_df.iterrows():
            parameter = row[column]["parameter"]
            value = row[column]["value"]
            quality_ini_dict.setdefault(parameter, []).append(value)

    # Add the extracted quality values as new columns and drop the original quality columns
    stockpiles_df = pd.concat(
        [stockpiles_df, pd.DataFrame(quality_ini_dict, index=stockpiles_df.index)],
        axis=1,
    ).drop(columns=quality_cols, errors="ignore")

    return stockpiles_df


def process_stockpiles_and_engines(
    stockpiles_df: pd.DataFrame, engines_df: pd.DataFrame
) -> pd.DataFrame:
    """Process stockpiles and engines.

    This function assigns engines to stockpiles and extracts initial quality
    values from the stockpiles DataFrame. It takes two DataFrames as input:
    one containing information about stockpiles and the other containing
    information about engines. The function processes these DataFrames to
    produce a new DataFrame that includes the assigned engines and the
    extracted quality values.

    Parameters
    ----------
    stockpiles_df : pd.DataFrame
        A `pandas.DataFrame` containing stockpile information.
    engines_df : pd.DataFrame
        A `pandas.DataFrame` containing engine information.

    Returns
    -------
    pd.DataFrame
        Processed `stockpiles_df` with engines assigned and quality values extracted.
    """
    stockpiles_df = assign_engines_to_stockpiles(stockpiles_df, engines_df)
    stockpiles_df = extract_quality_ini_values(stockpiles_df)
    return stockpiles_df


def travel_time(grp: pd.DataFrame) -> pd.DataFrame:
    """Calculate the travel time between consecutive events within a group.

    This function computes the time difference between the end of one event
    and the start of the next event in a grouped DataFrame. If the group
    contains only one event, the travel time is set to 0. The function
    expects the DataFrame to have 'start_time' and 'end_time' columns, and
    it should be pre-grouped by a relevant key. The resulting DataFrame will
    contain a single column 'travel_time', which represents the calculated
    travel times.

    Args:
        grp (pd.DataFrame): A `pandas.DataFrame` containing at least 'start_time' and 'end_time'

    Returns:
        pd.DataFrame: A `pandas.DataFrame` with a single column 'travel_time',
        containing the calculated travel times between consecutive events.
        The index of the returned DataFrame matches the input DataFrame.
    """
    if len(grp) == 1:
        return pd.DataFrame(
            {"travel_time": [grp["start_time"].values[0]]}, index=grp.index
        )
    grp = grp.sort_values(["end_time"])
    end_time = None
    res = []
    for _, row in grp.iterrows():
        _end_time = row["end_time"]
        if end_time is None:
            res.append(row["start_time"])
        else:
            res.append(row["start_time"] - end_time)
        end_time = _end_time
    return pd.DataFrame({"travel_time": res}, index=grp.index)


def json_input_output_to_excel(
    json_input_path: str | Path,
    json_output_path: str | Path,
    excel_path: str | Path | None = None,
):
    """Convert JSON input and output data to an Excel file.

    This function reads JSON data from specified input and output paths,
    processes the data into various DataFrames, and then saves the results
    into an Excel file. If an Excel path is not provided, it generates a
    default path based on the output JSON path. The function handles
    multiple aspects of the data, including stockpiles, engines, and quality
    metrics, and organizes them into a structured format suitable for
    analysis.

    Parameters
    ----------
    json_input_path : str | Path
        The path to the input JSON file containing instance data.
    json_output_path : str | Path 
        The path to the output JSON file containing results.
    excel_path : str | Path | None
        The path where the Excel file will be saved.
        If None, a default path will be generated. Defaults to None.

    Returns
    -------
    tuple
        A tuple containing two DataFrames:
          - operations_df: DataFrame with processed operations data.
          - outputs_df_out: DataFrame with processed output data.
    """

    check_is_file(json_input_path, json_output_path)

    if excel_path is None:
        excel_filename = Path(json_output_path).with_suffix(".xlsx").name
        excel_folder_path = Path(json_output_path).parent.parent.joinpath("excel")
        excel_path = str(excel_folder_path.joinpath(excel_filename))
    else:
        excel_folder_path = Path(excel_path).parent

    excel_folder_path.mkdir(exist_ok=True, parents=True)

    # Load JSON files
    # Input file
    with open(json_input_path) as fh:
        instance_data = json.load(fh)

    # Output file
    with open(json_output_path) as fh:
        output_data = json.load(fh)

    info_df = pd.DataFrame(
        [instance_data["info"]], columns=["Instance_Name", "Capacity", "Yard"]
    )
    engines_df = pd.DataFrame(instance_data["engines"])
    stockpiles_df = pd.DataFrame(instance_data["stockpiles"])
    stockpiles_quality_df = pd.json_normalize(
        stockpiles_df.pop("qualityIni"), sep="_"
    ).add_prefix("qualityIni_")
    stockpiles_df = pd.concat([stockpiles_df, stockpiles_quality_df], axis=1)
    stockpiles_df = process_stockpiles_and_engines(stockpiles_df, engines_df)

    inputs_df = pd.DataFrame(instance_data["inputs"])
    inputs_quality_df = pd.json_normalize(inputs_df.pop("quality"), sep="_").add_prefix(
        "quality_"
    )
    inputs_df = pd.concat([inputs_df, inputs_quality_df], axis=1)

    # Convert instance_1.json to DataFrames
    outputs_df = pd.DataFrame(instance_data["outputs"])
    outputs_quality_df = pd.json_normalize(
        outputs_df.pop("quality"), sep="_"
    ).add_prefix("quality_")
    outputs_df = pd.concat([outputs_df, outputs_quality_df], axis=1)

    # Explode the outputs_df
    outputs_df = explode_quality_rows(outputs_df, quality_col_prefix="quality_").drop(
        columns=["time"], errors="ignore"
    )
    distances_travel_df = pd.DataFrame(instance_data["distancesTravel"])
    time_travel_df = pd.DataFrame(instance_data["timeTravel"])

    time_travel_df.columns += 1
    time_travel_df.index += 1

    distances_travel_df.columns += 1
    distances_travel_df.index += 1

    from_to_list = []
    for col in time_travel_df.columns:
        for idx in time_travel_df.index:
            from_to_list.append([f"{col} -> {idx}", time_travel_df.loc[idx, col]])
    from_to_df = pd.DataFrame(from_to_list, columns=["from_to", "duration"])

    engines_df[from_to_df["from_to"].to_list()] = -1
    engines_df[from_to_df["from_to"].to_list()] = engines_df[
        from_to_df["from_to"].to_list()
    ].astype(float)

    for idx, row in engines_df.iterrows():
        yards = row["yards"]
        rail = row["rail"]
        stockpiles = []
        for _, stockpile_row in stockpiles_df.iterrows():
            if stockpile_row["yard"] in yards and rail in stockpile_row["rails"]:
                stockpiles.append(stockpile_row["id"])
        for start_stockpile in stockpiles:
            for end_stockpile in stockpiles:
                column_name = f"{start_stockpile} -> {end_stockpile}"
                duration = from_to_df.loc[
                    from_to_df["from_to"] == column_name, "duration"
                ].values[0]
                engines_df.loc[engines_df.index == idx, column_name] = duration

    engines_df[from_to_df["from_to"].to_list()] = engines_df[
        from_to_df["from_to"].to_list()
    ].replace(-1, "")

    objective_df = pd.DataFrame(
        [{"Objective": output_data["objective"], "Gap": output_data["gap"][0]}]
    )
    stacks_df = pd.DataFrame(output_data["stacks"])
    reclaims_df = pd.DataFrame(output_data["reclaims"])

    outputs_df_out = pd.DataFrame(output_data["outputs"])
    outputs_quality_df_out = (
        pd.json_normalize(outputs_df_out.pop("quality"), sep="_")
        .add_prefix("quality_")
    )

    outputs_df_out = pd.concat([outputs_df_out, outputs_quality_df_out], axis=1)
    outputs_df_out = explode_quality_rows(outputs_df_out, quality_col_prefix="quality_")

    if not stacks_df.empty:
        stacks_df["end_time"] = stacks_df["start_time"] + stacks_df["duration"]
        stacks_df["operation"] = "stack"

    if not reclaims_df.empty:
        reclaims_df["end_time"] = reclaims_df["start_time"] + reclaims_df["duration"]
        reclaims_df["operation"] = "reclaim"

    operations_df = (
        pd.concat([xdf for xdf in [stacks_df, reclaims_df] if not xdf.empty])
        .astype({"weight": int})
        .sort_values(["engine", "start_time"])
        .assign(
            travel_time=lambda xdf: (
                xdf.groupby("engine", as_index=False).apply(travel_time)
            )["travel_time"].reset_index(level=0, drop=True)
        )
    )

    stockpiles_final_df = (
        stockpiles_df.merge(
            operations_df.rename(columns={"weight": "weightFinal"})
            .groupby("stockpile")["weightFinal"]
            .sum(),
            left_on="id",
            right_index=True,
            how="left",
        )
        .fillna({"weightFinal": 0})
        .astype({"weightFinal": int})
        .assign(weightFinal=lambda xdf: xdf["weightIni"] - xdf["weightFinal"])
    )

    quality_cols = stockpiles_df.columns.intersection(
        ["Fe", "SiO2", "Al2O3", "P", "+31.5", "-6.3"]
    ).to_list()

    operations_df = operations_df.merge(
        stockpiles_df.rename(columns={"id": "stockpile"})[
            ["stockpile", "weightIni", *quality_cols]
        ],
        on="stockpile",
        how="left",
    ).assign(weightFinal=lambda xdf: xdf["weightIni"] - xdf["weight"])

    final_output_row = [
        operations_df["weight"].sum(),
        "output",
        operations_df["engine"].unique().tolist(),
        operations_df["start_time"].min(),
        operations_df["end_time"].max(),
        1,
        operations_df["end_time"].max(),
        "output_stack",
        operations_df["travel_time"].sum(),
        operations_df["weightFinal"].sum(),
    ]

    for quality_col in quality_cols:
        final_output_row.append(
            (
                (
                    operations_df["weight"]
                    * operations_df[quality_col]
                    / operations_df["weight"].sum()
                ).sum()
            )
        )
    operations_df = pd.concat(
        [
            operations_df,
            pd.DataFrame(
                {
                    col: [value]
                    for col, value in zip(operations_df.columns, final_output_row)
                }
            ),
        ],
        axis=0,
    )

    required_weight = operations_df.loc[
        operations_df["stockpile"] == "output", "weight"
    ].values[0]
    infos_gerais = pd.DataFrame(
        {"Variável": ["Peso Carregamento"], "Valor": [required_weight]}
    )

    engines_yards = (
        stockpiles_final_df[["yard", "engines"]]
        .astype({"engines": str})
        .drop_duplicates()
        .assign(
            engines=lambda xdf: xdf["engines"]
            .str.replace("[", "")
            .str.replace("]", "")
            .str.replace(" ", "")
            .str.replace(",", "")
            .apply(list)
        )
    )
    all_engines = list(
        sorted(
            set([engine for engines in engines_yards["engines"] for engine in engines])
        )
    )
    for engine in all_engines:
        engines_yards[f"Veículo {engine}"] = engines_yards["engines"].apply(
            lambda value: "x" if engine in value else ""
        )

    engines_yards = engines_yards.drop(columns=["engines"]).rename({"yard": "Área"})

    rename_dict = {
        "id": "ID",
        "yard": "Área",
        "weightIni": "Quantidade (ton)",
    }
    final_cols = [*list(rename_dict.values()), *quality_cols]
    stockpiles_final_df = stockpiles_final_df.rename(columns=rename_dict)[final_cols]

    load_rates = (
        engines_df[["id", "speedReclaim"]]
        .astype({"speedReclaim": int})
        .rename(columns={"id": "Veículo", "speedReclaim": "Taxa (ton/min)"})
    )

    from_to_cols = [col for col in engines_df.columns if "->" in col]
    travel_times_dict = {
        "De": [],
        "Para": [],
    }
    for from_to in from_to_cols:
        from_stockpile, to_stockpile = from_to.split(" -> ")
        travel_times_dict["De"].append(from_stockpile)
        travel_times_dict["Para"].append(to_stockpile)
        for engine in all_engines:
            vehicle_travel_times = travel_times_dict.get(f"Veículo {engine}", [])
            engine_row = engines_df.loc[engines_df["id"] == int(engine)]
            time_travel = engine_row[from_to].values[0]
            vehicle_travel_times.append(time_travel)
            travel_times_dict[f"Veículo {engine}"] = vehicle_travel_times

    travel_times_df = pd.DataFrame(travel_times_dict).astype({"De": int, "Para": int})

    rename_dict = {
        "engine": "Veículo",
        "stockpile": "Pilha",
        "weightIni": "Peso Inicial Pilha",
        "weightFinal": "Peso Final Pilha",
        "weight": "Carregamento (ton)",
        "start_time": "Início",
        "end_time": "Fim",
        "duration": "Tempo Carregamento",
        "travel_time": "Tempo Deslocamento",
    }
    operations_df = operations_df.rename(columns=rename_dict)[
        [*list(rename_dict.values()), *quality_cols]
    ]
    operations_df[quality_cols] = operations_df[quality_cols].round(2)
    operations_df = operations_df.fillna("")

    outputs_df_out["check"] = np.where(
        (outputs_df_out["value"] >= outputs_df_out["minimum"])
        & (outputs_df_out["value"] <= outputs_df_out["maximum"]),
        True,
        False,
    )
    # outputs_df_out = outputs_df_out.loc[~outputs_df_out["parameter"].isin(["Dummy"]), :]
    rename_dict = {
        "parameter": "Elemento",
        "value": "Valor",
        "minimum": "Mínimo",
        "maximum": "Máximo",
        "goal": "Meta",
        "check": "Check",
    }
    outputs_df_out = outputs_df_out.rename(columns=rename_dict)[
        list(rename_dict.values())
    ]
    return operations_df, outputs_df_out


def run_pyblend_command(
    input_json: str, output_json: str, algorithm: str = "lahc"
) -> None:
    """Runs the pyblend command with the specified input and output JSON files.

    This function constructs and executes a command to run the pyblend tool
    using the provided input and output JSON file paths. It allows for an
    optional algorithm parameter to specify which algorithm to use during
    execution. If the command fails to execute successfully, a RuntimeError
    is raised.

    Args:
        input_json (str): The input JSON file path.
        output_json (str): The output JSON file path.
        algorithm (str?): The algorithm to be used. Defaults to "lahc".

    Raises:
        RuntimeError: If the command fails to execute successfully.
    """
    pyblend_path = str(Path(__file__).parent.joinpath("./pyblend").resolve())
    command = [
        "python",
        pyblend_path,
        input_json,
        output_json,
        "-algorithm",
        algorithm,
        # "-maxiters",
        # "25000",
        # " -constructive",
        # "premodel"
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logging.info("Command executed successfully. Output:\n%s", result.stdout)
    except subprocess.CalledProcessError as e:
        logging.error("Command failed with error:\n%s", e.stderr)
        raise RuntimeError(f"Command failed: {e.stderr}")


class ExcelDataExtractor:
    """Class for extracting data from Excel sheets using xlwings.

    Methods
    -------
    extract_dataframe
        Extracts a DataFrame from a specified range in the sheet.
    """

    def __init__(self, workbook_path: str, sheet_name: str):
        """Initialize the ExcelDataExtractor with workbook path and sheet name.

        Parameters
        ----------
        workbook_path : str
            The path to the Excel workbook.
        sheet_name : str
            The name of the sheet from which data is extracted.
        """
        self.wb = xw.Book(workbook_path)
        self.sheet = self.wb.sheets[sheet_name]

    def extract_dataframe(self, range: str, expand: bool = True) -> pd.DataFrame:
        """Extract a DataFrame from a specified range in the sheet.

        This function retrieves data from a given cell range in a sheet and
        returns it as a pandas DataFrame. The user has the option to expand the
        range into a table format, which can be useful for structured data
        representation. If the expand parameter is set to True, the function
        will treat the specified range as a table; otherwise, it will return the
        data as is.

        Parameters
        ----------
        range : str
            The cell range to start extracting data from.
        expand : bool, default=True
            Whether to expand the range to a table.

        Returns
        -------
        pd.DataFrame
            Extracted data as a pandas DataFrame.
        """
        return (
            self.sheet[range]
            .options(pd.DataFrame, expand="table" if expand else None)
            .value.reset_index()
        )


class StockpileProcessor:
    """Class for processing stockpile data.

    Methods
    -------
    process
        Processes and merges stockpile and yard data.
    """

    @staticmethod
    def process(
        stockpiles: pd.DataFrame, yards: pd.DataFrame, rename_dict: Dict[str, str]
    ) -> pd.DataFrame:
        """Processes and merges stockpile and yard data.

        This function takes in two pandas DataFrames: one containing stockpile
        information and the other containing yard information. It renames the
        columns of the stockpile DataFrame based on the provided dictionary and
        converts the renamed columns to integers. After processing the yards
        DataFrame, it merges the two DataFrames on the 'yard' column using a
        left join, resulting in a combined DataFrame that includes both
        stockpile and yard data.

        Parameters
        ----------
        stockpiles : pd.DataFrame
            A `pandas.DataFrame` containing stockpile information.
        yards : pd.DataFrame
            A `pandas.DataFrame` containing yard information.
        rename_dict : Dict[str, str]
            Dictionary for renaming columns.

        Returns
        -------
        pd.DataFrame
            Merged DataFrame of stockpiles and yards. DataFrame of stockpiles and yards.
        """
        stockpiles = stockpiles.rename(columns=rename_dict).astype(
            {col: int for col in rename_dict.values()}
        )
        yards = StockpileProcessor._process_yards(yards)
        return stockpiles.merge(yards, on="yard", how="left")

    @staticmethod
    def _process_yards(yards: pd.DataFrame) -> pd.DataFrame:
        """Processes the yard data to extract rails information.

        This function takes a pandas DataFrame containing yard information,
        identifies columns that represent engine data by checking for numeric
        characters in their names, and renames these columns to retain only the
        numeric part. It then creates a new column that lists the indices of the
        non-null values in the engine columns, representing the rails associated
        with each yard. Finally, it returns a DataFrame with the processed data,
        excluding the original engine columns.

        Parameters
        ----------
        yards : pd.DataFrame
            A `pandas.DataFrame` containing yard information.

        Returns
        -------
        pd.DataFrame
          Processed DataFrame with rails information.
        """
        engine_cols = [
            col for col in yards.columns if any(ch.isnumeric() for ch in col)
        ]
        rename_dict = {
            col: "".join(ch for ch in col if ch.isnumeric()) for col in engine_cols
        }
        engine_ids = list(rename_dict.values())

        yards = yards.rename(columns=rename_dict).astype({"yard": int})
        yards["rails"] = yards[engine_ids].apply(
            lambda row: [idx for idx, value in enumerate(row, 1) if pd.notna(value)],
            axis=1,
        )
        return yards.drop(columns=engine_ids)


class TravelSpeedProcessor:
    """
    Class for processing travel speed data.

    Methods
    -------
    process
        Processes travel speed data into a list of travel times.
    """

    @staticmethod
    def process(
        travel_speed: pd.DataFrame, rename_dict: Dict[str, str]
    ) -> List[List[float]]:
        """Process travel speed data into a list of travel times.

        This function takes a DataFrame containing travel speed data and a
        dictionary for renaming columns. It processes the data to calculate
        travel times between locations, handling missing values by filling them
        with available speed data. The resulting travel times are returned as a
        nested list, where each sublist represents travel times from a specific
        origin to various destinations.

        Parameters
        ----------
        travel_speed : pd.DataFrame
            A `pandas.DataFrame` containing travel speed data.
        rename_dict : Dict[str, str]
            Dictionary for renaming columns.

        Returns
        -------
        List[List[float]]
            Nested list representing travel times between locations.
        """
        travel_speed = travel_speed.rename(columns=rename_dict)
        stockpiles_from_to_cols = list(rename_dict.values())
        speed_cols = travel_speed.columns.difference(stockpiles_from_to_cols)

        travel_speed["travel_time"] = travel_speed[speed_cols[0]]
        for col in speed_cols[1:]:
            travel_speed["travel_time"] = travel_speed["travel_time"].fillna(
                travel_speed[col]
            )

        return (
            travel_speed.drop(columns=speed_cols)
            .pivot_table(index=["from"], columns=["to"], values=["travel_time"])
            .values.tolist()
        )


class InstanceDataBuilder:
    """Class for building the instance data JSON structure.

    Methods
    -------
    build_instance_data
        Builds the instance data structure.
    """

    @staticmethod
    def build_instance_data(
        stockpiles: pd.DataFrame,
        engines: pd.DataFrame,
        travel_times: List[List[float]],
        inputs: List[Dict[str, Any]],
        outputs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Builds the instance data structure.

        This function constructs a comprehensive data structure that includes
        information about stockpiles, engines, input data, output data, and
        travel times. It processes the provided stockpile and engine data from
        pandas DataFrames and organizes them into a dictionary format suitable
        for further analysis or processing. The function iterates through the
        stockpile and engine DataFrames to extract relevant attributes and
        compiles them into a structured format.

        Parameters
        ----------
        stockpiles : pd.DataFrame
            A `pandas.DataFrame` containing stockpile information.
        engines : pd.DataFrame
            A `pandas.DataFrame` containing engine information.
        travel_times : List[List[float]]
            Nested list representing travel times between locations.
        inputs : List[Dict[str, Any]])
            List of dictionaries representing input data.
        outputs : List[Dict[str, Any]]): List of dictionaries representing output data.

        Returns:
            Dict[str, Any]: The constructed instance data structure.
        """
        instance_data = {
            "info": ["Instance_Interactive", 1, 1],
            "stockpiles": [],
            "engines": [],
            "inputs": inputs,
            "outputs": outputs,
            "distancesTravel": travel_times,
            "timeTravel": travel_times,
        }

        all_fe_the_same = stockpiles["Fe"].nunique() == 1
        all_sio2_the_same = stockpiles["SiO2"].nunique() == 1
        all_al2o3_the_same = stockpiles["Al2O3"].nunique() == 1
        all_p_the_same = stockpiles["P"].nunique() == 1
        all_the_same = (stockpiles[["Al2O3", "Fe", "SiO2", "P"]].round(2).nunique() == 1).all()
        all_w_the_same = stockpiles["weightIni"].nunique() == 1

        if all_w_the_same:
            stockpiles["weightIni"].iloc[-1] = 0

        stockpiles["Dummy"] = np.random.randint(1, 100, stockpiles.shape[0])

        for _, row in stockpiles.iterrows():
            sio2 = row["SiO2"]
            fe = row["Fe"]
            al2o3 = row["Al2O3"]
            p = row["P"]
            quality_ini = [
                {"parameter": "Fe", "value": float(round(fe, 4))},
                {"parameter": "SiO2", "value": float(round(sio2, 4))},
                {"parameter": "Al2O3", "value": float(round(al2o3, 4))},
                {"parameter": "P", "value": float(round(p, 4))},
                {"parameter": "Dummy", "value": float(row["Dummy"])}
            ]

            sp = {
                "id": int(row["id"]),
                "position": int(row["id"]) - 1,
                "yard": int(row["yard"]),
                "rails": [int(r) for r in row["rails"]],
                "capacity": int(row["weightIni"]),
                "weightIni": int(row["weightIni"]),
                "qualityIni": quality_ini,
            }
            instance_data["stockpiles"].append(sp)

        all_yards = [int(y) for y in stockpiles["yard"].drop_duplicates().to_list()]

        for _, row in engines.iterrows():
            eng = {
                "id": int(row["id"]),
                "speedStack": 0.0,
                "speedReclaim": int(row["speedReclaim"]),
                "posIni": int(row["id"]),
                "rail": int(row["id"]),
                "yards": all_yards,
            }
            instance_data["engines"].append(eng)

        instance_data["outputs"][0]["quality"].append(
            {
                "parameter": "Dummy",
                "minimum": 0,
                "maximum": 100,
                "goal": float(stockpiles["Dummy"].mean()),
                "importance": 1,
            }
        )
        return instance_data


def update_excel_sheets(operations_df: pd.DataFrame, outputs_df_out: pd.DataFrame,
                        excel_file: str) -> None:
    """Update the Excel sheets with the new data.

    This function updates specified Excel sheets with new data from provided
    DataFrames. It clears the existing contents of the sheets and writes the
    new data into them. The function also formats the Excel sheets for
    better readability and adjusts the columns to fit the content.

    Parameters
    ----------
    operations_df : pd.DataFrame
      A `pandas.DataFrame` containing operation results.
    outputs_df_out : pd.DataFrame
      A `pandas.DataFrame` containing output check results.
    excel_file : str
      Path to the Excel file to be updated.
    """
    wb = xw.Book(excel_file)
    resultados_sheet = wb.sheets['Resultados']
    check_res_sheet = wb.sheets['Check Restrições']

    # Clear existing data
    resultados_sheet.clear_contents()
    check_res_sheet.clear_contents()

    # Write new data
    operations_df = operations_df.fillna('')
    operations_df.iloc[-1, 0] = operations_df.iloc[0, -1].astype(str)
    operations_df = operations_df.set_index('Pilha')

    resultados_sheet.range("A1").value = operations_df
    format_excel_sheet(resultados_sheet)
    format_integer_columns(resultados_sheet)
    autofit_columns_from_sheet(resultados_sheet)

    check_res_sheet.range("A1").value = outputs_df_out.fillna('').set_index(
        outputs_df_out.columns[0]
    )


def generate_gantt_chart(operations_df, sheet):
    """Generate a Gantt chart from operations data.

    This function takes a DataFrame containing operations data and generates
    a Gantt chart visualizing the start and end times of each operation. It
    processes the time data to convert it into a format suitable for
    plotting, and then creates a horizontal bar chart where each bar
    represents an operation. The chart includes annotations for additional
    information about each operation.

    Parameters
    ----------
    operations_df : pd.DataFrame
        A DataFrame containing operations data
          with columns such as 'Início', 'Fim', 'Veículo', and 'Pilha'.
    sheet : object
         An object representing the Excel sheet where the Gantt chart will be added.
    """

    operations_df = operations_df.iloc[:-1]
    operations_df["Início"] = pd.to_timedelta(operations_df["Início"], unit="m")
    operations_df["Fim"] = pd.to_timedelta(operations_df["Fim"], unit="m")
    operations_df["Tempo Deslocamento (min)"] = (
            pd.to_timedelta(operations_df["Tempo Deslocamento"], unit="m").dt.total_seconds()
            / 60
    )

    # Convert the timedeltas to a more plot-friendly format by using hours as float
    operations_df["Início (min)"] = operations_df["Início"].dt.total_seconds() / 60
    operations_df["Fim (min)"] = operations_df["Fim"].dt.total_seconds() / 60
    operations_df["Início (min)"] -= operations_df["Tempo Deslocamento (min)"]

    # Create the Gantt plot with additional annotations
    fig, ax = plt.subplots(figsize=(9, 3))

    # Plot each operation and add text annotations
    for idx, row in operations_df.iterrows():
        ax.barh(
            row["Veículo"],
            row["Fim (min)"] - row["Início (min)"],
            left=row["Início (min)"],
        )
        # Calculate the position for the text
        mid_point = (row["Início (min)"] + row["Fim (min)"]) / 2
        label = f'SP: {int(row["Pilha"])}'
        ax.text(
            mid_point,
            row["Veículo"],
            label,
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            usetex=False,
        )

    # Formatting the plot
    ax.set_yticks([1, 2])
    ax.set_ylim(0.5, None)
    ax.set_xlabel("Tempo (minutos)")
    ax.set_ylabel("Veículo", fontsize=16)
    ax.set_title("Operações")

    sheet.pictures.add(
        fig,
        name='Gantt',
        update=True,
        anchor=sheet.range((sheet.used_range.last_cell.row + 2, 1)),
    )


def main(
    excel_filepath: str = "out_interactive.xlsm",
    instance_json_path: str = "./tests/instance_interactive.json",
    output_json_path: str = "./out/json/out_interactive.json",
):
    """
    Run the main process for extracting data from an Excel file and generating outputs.

    This function orchestrates the extraction of data from a specified Excel
    file, processes the data to build an instance data structure, and writes
    the results to a JSON file. It also executes a command to run a PyBlend
    operation and updates the Excel sheets with the results. The function
    handles various data types including stockpiles, yards, engines, and
    travel speeds, and generates a Gantt chart based on the operations
    extracted.

    Parameters
    ----------
    excel_filepath : str, default="out_interactive.xlsm"
        The path to the Excel file to be processed.
    instance_json_path : str, default="./tests/instance_interactive.json"
        The path where the instance JSON file will be saved.
    output_json_path : str, default="./out/json/out_interactive.json"
        The path where the output JSON file will be saved.
    """

    # excel_filepath = str(Path(excel_filepath).resolve())
    instance_json_path = str(Path(__file__).parent.joinpath(instance_json_path).resolve())
    output_json_path = str(Path(__file__).parent.joinpath(output_json_path).resolve())

    # Initialize the ExcelDataExtractor with workbook path and sheet name
    extractor = ExcelDataExtractor(excel_filepath, "Inputs")

    # Extract data from the Excel sheet
    stockpiles_df = extractor.extract_dataframe("D2")
    yards_df = extractor.extract_dataframe("L2")
    engines_df = (
        extractor.extract_dataframe("P2")
        .rename(columns={"Veículo": "id", "Taxa (ton/min)": "speedReclaim"})
        .astype(int)
    )
    travel_speed_df = extractor.extract_dataframe("S2")

    # Process stockpile and yard data
    stockpiles = StockpileProcessor.process(
        stockpiles_df,
        yards_df,
        {"ID": "id", "Área": "yard", "Quantidade (ton)": "weightIni"},
    )
    # Process travel speed data
    travel_times = TravelSpeedProcessor.process(
        travel_speed_df, {"De": "from", "Para": "to"}
    )

    output_info = extractor.extract_dataframe("A2")
    outputs = [
        {
            "id": 1,
            "destination": 1,
            "weight": int(output_info.iloc[0, 1]),
            "quality": [
                {
                    "parameter": "Fe",
                    "minimum": 60,
                    "maximum": 100,
                    "goal": 65,
                    "importance": 10,
                },
                {
                    "parameter": "SiO2",
                    "minimum": 2.8,
                    "maximum": 5.8,
                    "goal": 5.8,
                    "importance": 10,
                },
                {
                    "parameter": "Al2O3",
                    "minimum": 2.5,
                    "maximum": 4.9,
                    "goal": 4.9,
                    "importance": 10,
                },
                {
                    "parameter": "P",
                    "minimum": 0.05,
                    "maximum": 0.07,
                    "goal": 0.07,
                    "importance": 10,
                },
            ],
            "time": 600,
        }
    ]

    inputs = [
        {
            "id": 1,
            "weight": 0.0,
            "quality": [
                {"parameter": "Fe", "value": 60},
                {"parameter": "SiO2", "value": 1.5},
                {"parameter": "Al2O3", "value": 0.8},
                {"parameter": "P", "value": 1.0},
            ],
            "time": 8.0,
        }
    ]

    # Build the instance data structure
    instance_data = InstanceDataBuilder.build_instance_data(
        stockpiles, engines_df, travel_times, inputs, outputs
    )

    # Write the data to a JSON file
    with open(instance_json_path, "w") as json_file:
        json.dump(instance_data, json_file, indent=2)

    run_pyblend_command(instance_json_path, output_json_path)

    operations_df, outputs_df_out = json_input_output_to_excel(
        instance_json_path,
        output_json_path,
    )
    update_excel_sheets(operations_df, outputs_df_out, excel_filepath)
    extractor = ExcelDataExtractor(excel_filepath, "Resultados")
    operations_df = extractor.extract_dataframe("A1")
    sheet = extractor.sheet

    generate_gantt_chart(operations_df, sheet)


if __name__ == "__main__":
    main(
        "out_interactive.xlsm",
        "./tests/instance_interactive.json",
        "./out/json/out_interactive.json",
    )
