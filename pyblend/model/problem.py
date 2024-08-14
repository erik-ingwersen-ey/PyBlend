from typing import List, Union

import ujson

from pyblend.config import Engines, Inputs, Outputs, Stockpiles, Travels
from pyblend.model.classes import Engine, Input, Output, Quality, Request, Stockpile


class Problem:
    """
    Ore Mixing Problem and Machine Scheduling Problem class.

    This class is used to build the Ore Mixing and the Machine Scheduling Problems
    from a JSON file.
    """

    def __init__(self: "Problem", instance_path: str):
        """Build a new Problem from a file.

        Parameters
        ----------
        instance_path : str
            The instance path.
        """

        with open(instance_path, "r") as file:
            data = ujson.load(file)

        self._info: List[Union[str, int]] = data["info"]

        self._stockpiles: Stockpiles = [
            Stockpile(
                data["id"],
                data["position"],
                data["yard"],
                data["rails"],
                data["capacity"],
                data["weightIni"],
                [Quality(*q.values()) for q in data["qualityIni"]],
            )
            for data in data["stockpiles"]
        ]

        self._engines: Engines = [
            Engine(
                data["id"],
                data["speedStack"],
                data["speedReclaim"],
                data["posIni"],
                data["rail"],
                data["yards"],
            )
            for data in data["engines"]
        ]

        self._inputs: Inputs = [
            Input(
                data["id"],
                data["weight"],
                [Quality(*q.values()) for q in data["quality"]],
                data["time"],
            )
            for data in data["inputs"]
        ]

        self._outputs: Outputs = [
            Output(
                data["id"],
                data["destination"],
                data["weight"],
                [Request(*q.values()) for q in data["quality"]],
                data["time"],
            )
            for data in data["outputs"]
        ]

        self._distances_travel: Travels = data["distancesTravel"]
        self._time_travel: Travels = data["timeTravel"]

    @property
    def info(self: "Problem") -> List[Union[str, int]]:
        """
        List with the instance name and the omega values for the linear model.

        Returns
        -------
        List[Union[str, int]]
            List with the instance name and the omega values for the linear model.
        """
        return self._info

    @info.setter
    def info(self: "Problem", value: List[Union[str, int]]) -> None:
        self._info = value

    @property
    def stockpiles(self: "Problem") -> Stockpiles:
        """
        List or stockpile data.

        Returns
        -------
        List[Stockpile]
            List with stockpile data.
        """
        return self._stockpiles

    @stockpiles.setter
    def stockpiles(self: "Problem", value: Stockpiles) -> None:
        self._stockpiles = value

    @property
    def engines(self: "Problem") -> Engines:
        """
        List with the engine data.

        Returns
        -------
        List[Engine]
            List with engine data.
        """
        return self._engines

    @engines.setter
    def engines(self: "Problem", value: Engines) -> None:
        self._engines = value

    @property
    def inputs(self: "Problem") -> Inputs:
        """
        List with the ore input data.

        Returns
        -------
        List[Input]
            List with ore input data.
        """
        return self._inputs

    @inputs.setter
    def inputs(self: "Problem", value: Inputs) -> None:
        self._inputs = value

    @property
    def outputs(self: "Problem") -> Outputs:
        """
        list with the ore output data.

        Returns
        -------
        List[Output]
            list with ore output data.
        """
        return self._outputs

    @outputs.setter
    def outputs(self: "Problem", value: Outputs) -> None:
        self._outputs = value

    @property
    def distances_travel(self: "Problem") -> Travels:
        """
        Matrix with the distances between each stockpile.

        Returns
        -------
        List[List[float]]
            Matrix with the distances between each stockpile.
        """
        return self._distances_travel

    @distances_travel.setter
    def distances_travel(self: "Problem", value: Travels) -> None:
        self._distances_travel = value

    @property
    def time_travel(self: "Problem") -> Travels:
        """
        Matrix with the time needed to travel from one stockpile to another.

        Returns
        -------
        List[List[float]]
            Matrix with the time needed to travel from one stockpile to another.
        """
        return self._time_travel

    @time_travel.setter
    def time_travel(self: "Problem", value: Travels) -> None:
        self._time_travel = value
