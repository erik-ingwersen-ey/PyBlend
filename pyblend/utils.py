from __future__ import annotations

import datetime
from pathlib import Path


def handle_input_path(input_path: str | Path) -> str:
    """Validate and resolve the path to the model input JSON file.

    This function checks whether the provided path points to a valid JSON
    file. If the input path is a relative path and the file isn't found in
    the current directory, the function searches up to two levels of parent
    directories to locate the file. If the file is found, its absolute path
    is returned. If the input path does not point to a JSON file or if the
    file cannot be found, appropriate exceptions are raised.

    Args:
        input_path (str or Path): The input path to check, which should point to a JSON file.

    Returns:
        str: The resolved absolute path to the JSON file.

    Raises:
        ValueError: If the input path doesn't point to a JSON file.
        FileNotFoundError: If the file can't be found after searching the current directory
    """
    input_path = Path(input_path)
    if input_path.suffix != ".json":
        raise ValueError(
            f"Input path should point to a JSON file. Value provided: '{input_path}'"
        )

    if input_path.parent == Path("."):
        curr_dir = Path.cwd()
        files_found = list(curr_dir.glob(str(input_path)))
        if len(files_found) > 0:
            return str(files_found[0])

        max_upper_dirs = 2
        while max_upper_dirs > 0:
            files_found = list(curr_dir.glob(f"**/{input_path}"))
            if len(files_found) > 0:
                return str(files_found[0])

            max_upper_dirs -= 1
            curr_dir = curr_dir.parent

    raise FileNotFoundError(f"Could not find file '{input_path}'")


def handle_output_path(output_path: str | Path) -> str:
    """Handle and resolve the output path for the problem output JSON file.

    This function takes an output path and ensures it is prepared to store a
    JSON file. If the path is a directory or lacks a file suffix, the
    function creates the necessary directories and generates a default JSON
    filename with the current timestamp. If the path specifies a file with a
    non-JSON extension, a `ValueError` is raised.

    Args:
        output_path (str or Path): The output path where the JSON file should be stored.

    Returns:
        str: The resolved output path as a string, pointing to a JSON file.

    Raises:
        ValueError: If the output path specifies a file with a non-JSON extension.

    Examples:
        Handle a directory path and generate a JSON file:
        
        >>> path = handle_output_path("output_directory/")
        >>> print(path)
        'output_directory/json/out-2024-08-08 14_30_00.json'
        
        Handle an output path with a JSON file specified:
        
        >>> path = handle_output_path("output_directory/result.json")
        >>> print(path)
        'output_directory/result.json'
    """
    output_path = Path(output_path)
    if output_path.suffix == "":
        output_path.mkdir(exist_ok=True, parents=True)
        if output_path.name != "json":
            output_path = output_path.joinpath("json").mkdir(exist_ok=True, parents=True)
        file_identifier = datetime.datetime.today().strftime("%Y-%m-%d %H_%M_%S")
        return str(output_path.joinpath("json").joinpath(f"out-{file_identifier}").with_suffix(".json"))

    if output_path.suffix != ".json":
        raise ValueError(
            f"Output path should point to a JSON file or directory. Value provided: '{output_path}'"
        )

    output_path.parent.mkdir(exist_ok=True, parents=True)
    return str(output_path)
