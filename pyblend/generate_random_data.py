from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List


def validate_parameters(output_weight, quality_params):
    """Validate the quality parameters and output weight.

    This function checks if the provided output weight is greater than zero
    and verifies that each quality parameter contains valid values for
    'minimum', 'maximum', 'goal', and 'importance'. It ensures that the goal
    is within the specified range of minimum and maximum values, and that
    the importance is a positive value. If any of these conditions are not
    met, a ValueError is raised with an appropriate message.

    Args:
        output_weight (float): The weight of the output to be validated.
        quality_params (list): A list of dictionaries containing quality
            parameters. Each dictionary must include 'minimum', 'maximum',
            'goal', and 'importance'.

    Raises:
        ValueError: If output_weight is less than or equal to 0.
        ValueError: If any quality parameter is missing required fields
            or if the values are not valid.
    """
    if output_weight <= 0:
        raise ValueError("Output weight must be greater than 0.")

    for param in quality_params:
        minimum = param.get("minimum")
        maximum = param.get("maximum")
        goal = param.get("goal")
        importance = param.get("importance")

        if minimum is None or maximum is None or goal is None or importance is None:
            raise ValueError(
                "All quality parameters must include 'minimum', 'maximum', 'goal', and 'importance'.")

        if not (minimum < goal <= maximum):
            raise ValueError(
                f"Invalid quality parameter values: minimum ({minimum}), goal ({goal}), maximum ({maximum}).")

        if importance <= 0:
            raise ValueError("Importance must be greater than 0.")


def generate_quality_values(quality_params):
    """Generate random quality values for stockpiles based on output quality
    parameters.

    This function takes a list of quality parameters, each containing a
    minimum and maximum value, and generates random quality values for
    stockpiles. For each parameter, a random value is generated within the
    specified range and rounded to two decimal places. The resulting quality
    values are returned as a list of dictionaries, where each dictionary
    contains the parameter name and its corresponding generated value.

    Args:
        quality_params (list): A list of dictionaries, each containing 'parameter', 'minimum',
            and 'maximum' keys.

    Returns:
        list: A list of dictionaries with generated quality values for each parameter.
    """
    stock_quality = []
    for param in quality_params:
        quality = {
            "parameter": param["parameter"],
            "value": round(random.uniform(param["minimum"], param["maximum"]), 2)
        }
        stock_quality.append(quality)
    return stock_quality


def generate_problem_data(num_stockpiles, num_engines, can_access_all, output_weight,
                          quality_params):
    """Generate problem data in JSON format based on the given parameters.

    This function creates a structured representation of stockpiles and
    engines for a problem instance. It validates the input parameters,
    generates stockpile data with random weights and quality values, and
    creates engine data with random attributes. Additionally, it computes
    distances and travel times between stockpiles to facilitate further
    processing.

    Args:
        num_stockpiles (int): The number of stockpiles to generate.
        num_engines (int): The number of engines to generate.
        can_access_all (bool): Indicates if all yards can be accessed by engines.
        output_weight (float): The required total output weight from the stockpiles.
        quality_params (list): A list of dictionaries containing quality parameters
            for the stockpiles.

    Returns:
        dict: A dictionary containing the generated problem data in JSON format.

    Raises:
        ValueError: If the sum of all stockpile weights is not greater than the
            specified output weight.
    """

    # Validate the output weight and quality parameters
    validate_parameters(output_weight, quality_params)

    # Create stockpiles with valid quality parameters
    stockpiles = []
    total_weight = 0

    for i in range(1, num_stockpiles + 1):
        weight_ini = int(
            round(
                random.uniform(
                    output_weight / num_stockpiles,
                    output_weight / (num_stockpiles - i + 1)
                ),
                0,
            )
        )
        total_weight += weight_ini
        stockpile = {
            "id": i,
            "position": i - 1,
            "yard": random.randint(1, num_stockpiles // 5 + 1),
            # Randomly assign to a yard
            "rails": [1, 2] if can_access_all else [random.randint(1, 2)],
            "capacity": weight_ini,  # Assuming capacity is double the initial weight
            "weightIni": weight_ini,
            "qualityIni": generate_quality_values(quality_params)
        }
        stockpiles.append(stockpile)

    # Ensure the total weight exceeds the required output weight
    if total_weight <= output_weight:
        raise ValueError(
            "Sum of all stockpile weights must be greater than the specified output weight.")

    # Generate engines
    engines = []
    for j in range(1, num_engines + 1):
        engine = {
            "id": j,
            "speedStack": 0.0,
            "speedReclaim": random.randint(2500, 4000),
            "posIni": random.randint(1, num_stockpiles),
            "rail": random.randint(1, 2),
            "yards": list(range(1, num_stockpiles // 5 + 2)) if can_access_all else [
                random.randint(1, num_stockpiles // 5 + 1)]
        }
        engines.append(engine)

    # Generate distances and travel times between stockpiles
    distances_travel = []
    time_travel = []
    for i in range(num_stockpiles):
        distances_row = []
        time_row = []
        for j in range(num_stockpiles):
            if i == j:
                distances_row.append(0.0)
                time_row.append(0.06)
            else:
                distance = round(random.uniform(0.05, 1.5), 2)
                distances_row.append(distance)
                time_row.append(round(distance / random.uniform(1, 2), 2))  #
                # Random time based on distance
        distances_travel.append(distances_row)
        time_travel.append(time_row)

    # Prepare the output JSON structure
    problem_data = {
        "info": ["Generated_Instance", 1000, 1],
        "stockpiles": stockpiles,
        "engines": engines,
        "inputs": [
            {
                "id": 1,
                "weight": 0.0,
                "quality": [{"parameter": param["parameter"], "value": param["goal"]}
                            for param in quality_params],
                "time": 0
            }
        ],
        "outputs": [
            {
                "id": 1,
                "destination": 1,
                "weight": output_weight,
                "quality": quality_params,
                "time": 0
            }
        ],
        "distancesTravel": distances_travel,
        "timeTravel": time_travel
    }

    return problem_data


def save_new_instance(
    data: dict,
    folder_path: str | Path,
    name_pattern: str = "instance_*.json",
):
    """Save a new instance of data to a specified folder with a unique
    filename.

    This function saves the provided data as a JSON file in the specified
    folder. It generates a unique filename based on the provided name
    pattern by enumerating existing files in the folder. If the folder does
    not exist, it will be created. The function ensures that the new
    filename does not conflict with existing files by incrementing the
    enumeration until a unique name is found.

    Args:
        data (dict): The data to be saved as a JSON file.
        folder_path (str | Path): The path to the folder where the file will be saved.
        name_pattern (str?): The pattern for naming the file. Defaults to "instance_*.json".

    Returns:
        None: This function does not return a value; it saves the data to a file.

    Raises:
        ValueError: If the provided folder path points to an existing file instead of a
            directory.
    """

    _folder_path = Path(folder_path)
    if _folder_path.is_file():
        raise ValueError(
            "The folder path must be a path to an existing or to be created "
            f"directory, not a file: '{folder_path}'"
        )
    _folder_path.mkdir(exist_ok=True, parents=True)
    files_enumerations = []
    for filepath in _folder_path.glob(name_pattern):
        filename = filepath.with_suffix("").name
        number = "".join(
            character for character in filename if character.isnumeric()
        )
        if number != "":
            files_enumerations.append(number)

    next_enumeration = 1

    if len(files_enumerations) > 0:
        files_enumerations = [int(number) for number in files_enumerations]
        last_enumeration = max(files_enumerations)
        next_enumeration = last_enumeration + 1

    new_filename = name_pattern.replace("*", str(next_enumeration))
    new_filepath = _folder_path.joinpath(new_filename).with_suffix(".json")

    while new_filepath.is_file():
        next_enumeration += 1
        new_filename = name_pattern.replace("*", str(next_enumeration))
        new_filepath = _folder_path.joinpath(new_filename).with_suffix(".json")

    with open(str(new_filepath), "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, allow_nan=False)

    print(f"Successfully saved generated data to: '{new_filepath}'")


def recursive_glob(
    initial_directory: Path,
    pattern: str,
    max_upper_dirs: int = 3,
    recursive: bool = True,
) -> Path | None:
    """Find the first directory matching a pattern, searching recursively.

    This function searches for directories that match a specified pattern
    starting from an initial directory. If no matches are found, it will
    move up to the parent directories (up to a specified limit) and continue
    the search. If the `recursive` flag is set to True and the pattern does
    not start with '**/', the function will prepend '**/' to the pattern to
    enable recursive searching.

    Args:
        initial_directory (Path): The starting directory for the search.
        pattern (str): The glob pattern to match against directory names.
        max_upper_dirs (int?): The maximum number of parent directories
            to search upwards. Defaults to 3.
        recursive (bool?): Whether to search recursively. Defaults to True.

    Returns:
        Path | None: The first matching directory as a Path object, or None if
        no matches are found.
    """

    if recursive and not pattern.startswith("**/"):
        pattern = f"**/{pattern}"

    matches = [path for path in initial_directory.glob(pattern) if path.is_dir()]
    while len(matches) == 0 and max_upper_dirs > 0:
        initial_directory = initial_directory.parent
        matches = [path for path in initial_directory.glob(pattern) if path.is_dir()]
        max_upper_dirs -= 1

    if len(matches) > 0:
        return matches[0]

    return None


def generate_and_save_random_input(
    num_stockpiles: int = 20,
    num_engines: int = 2,
    output_weight: int = 500_000,
    can_access_all: bool = True,
    quality_params: List[Dict[str, str | int | float]] | None = None,
    output_folder_path: str | Path | None = None,
):
    """Generate and save random input data for stockpiles and engines.

    This function generates random input data based on specified parameters
    such as the number of stockpiles, engines, and output weight. It also
    allows for the specification of quality parameters that define the
    acceptable ranges for various materials. If no quality parameters are
    provided, default values are used. The generated data is then saved to a
    specified output folder. If the output folder does not exist, it will be
    created.

    Args:
        num_stockpiles (int): The number of stockpiles to generate. Default is 20.
        num_engines (int): The number of engines to generate. Default is 2.
        output_weight (int): The total weight of the output data. Default is 500,000.
        can_access_all (bool): Flag indicating if all stockpiles can be accessed. Default is True.
        quality_params (List[Dict[str, str | int | float]] | None): A list of dictionaries defining
            quality parameters for the generated data. Each dictionary should
            contain keys for
            'parameter', 'minimum', 'maximum', 'goal', and 'importance'. If None,
            default parameters are used.
        output_folder_path (str | Path | None): The path to the folder where the generated data will be saved.
            If None, a default path will be determined.

    Raises:
        ValueError: If quality_params is provided but is not a list of dictionaries.
        FileNotFoundError: If output_folder_path is not specified and cannot be found.
    """

    if quality_params is None:
        _quality_params = [
            {"parameter": "Fe", "minimum": 60, "maximum": 100, "goal": 65, "importance": 10},
            {"parameter": "SiO2", "minimum": 2.8, "maximum": 5.8, "goal": 5.8, "importance": 1000},
            {"parameter": "Al2O3", "minimum": 2.5, "maximum": 4.9, "goal": 4.9, "importance": 100},
            {"parameter": "P", "minimum": 0.05, "maximum": 0.07, "goal": 0.07, "importance": 100}
        ]
        quality_params = _quality_params
    elif not isinstance(quality_params, list):
        raise ValueError("Quality parameters must be a list of dictionaries.")

    if output_folder_path is None:
        output_folder_path = recursive_glob(Path.cwd(), "**/tests")
    if output_folder_path is None:
        output_folder_path = recursive_glob(Path.cwd(), "**/pyblend")
        if output_folder_path is None:
            raise FileNotFoundError(
                "Output folder path not specified and could not be found."
            )
        else:
            output_folder_path = output_folder_path.parent.joinpath("tests")
            output_folder_path.mkdir(exist_ok=True, parents=True)

    generated_data = generate_problem_data(
        num_stockpiles=num_stockpiles,
        num_engines=num_engines,
        can_access_all=can_access_all,
        output_weight=output_weight,
        quality_params=quality_params,
    )

    save_new_instance(generated_data, output_folder_path)


if __name__ == "__main__":
    generate_and_save_random_input()
