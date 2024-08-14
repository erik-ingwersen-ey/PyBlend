from typing import Any, Dict, List, Optional, Tuple, Union

from pyblend.model.classes import Engine, Input, Output, Stockpile


# type aliases for the input data
Stockpiles = List[Stockpile]
Engines = List[Engine]
Inputs = List[Input]
Outputs = List[Output]
Travels = List[List[float]]
Information = Union[str, Stockpiles, Engines, Inputs, Outputs, Travels]
Data = Dict[str, Union[Information, Any]]

# type aliases for the output data
Jobs = List[Dict[str, Union[int, float]]]
Qualities = List[Dict[str, Union[str, int, float]]]
Deliveries = List[Dict[str, Union[float, Qualities]]]
Result = Dict[str, Union[str, float, Jobs, Deliveries]]

# type aliases for the linear model
Weights = Dict[str, List[float]]
Objective = Tuple[Optional[float], Weights, Weights]

# type aliases for the solver
Route = List[Tuple[float, int, int, str]]
Routes = List[List[Tuple[int, str]]]

# type aliases for instance generator data
QualityIni = List[Dict[str, Union[str, float]]]
Stock = List[Dict[str, Union[int, float, List[int], QualityIni]]]
Eng = List[Dict[str, Union[int, float, List[int]]]]
Inp = List[Dict[str, Union[int, float, QualityIni]]]
Out = List[Dict[str, Union[int, float, Qualities]]]
Instance = Dict[str, Union[str, Stock, Eng, Inp, Out, Travels]]

# type aliases for the .json file information
Info = Dict[str, Union[Result, Instance]]

# type aliases for the terminal parameters
Parameters = Dict[str, Union[str, int, float, Any]]
