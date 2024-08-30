from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List


def validate_parameters(output_weight, quality_params):
    """Validate the quality parameters and output weight."""
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
    """Generate random quality values for stockpiles based on output quality parameters."""
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
    """Generate the problem data in JSON format based on the given parameters."""

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
