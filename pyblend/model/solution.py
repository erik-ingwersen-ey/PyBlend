from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import ujson

from pyblend.config import (Deliveries, Jobs, Objective, Qualities, Result, Routes,
                            Weights)
from pyblend.model.classes import Request
from pyblend.model.problem import Problem


class Solution:
    """
    Represents a solution to the Machine Scheduling Problem and the Ore Mixing Problem.
    """

    def __init__(self: 'Solution', problem: Problem):
        """
        Initialize a new instance of the `Solution` class.

        Parameters
        ----------
        problem : Problem
            The problem instance to be solved.
        """

        self._problem: Problem = problem

        # Ore Mixing Problem
        self._objective: Optional[float] = None 
        self._weights: Weights | list = []
        self._inputs: Weights | list = []

        # Machine Scheduling Problem
        self._cost: float = float('inf')
        self._routes: Routes = [[] for _ in range(len(problem.engines))]
        self._start_time: List[float] = [0] * len(problem.engines)
        self._gap: List[float] = [1] * len(problem.outputs)
        self._stacks: Jobs = []
        self._reclaims: Jobs = []
        self._deliveries: Deliveries = []

        self._has_deliveries: bool = False

    def set_deliveries(self: 'Solution') -> None:
        """
        Save the output data for each order, including the total mass ordered,
        initiation time, duration, and quality of each ore delivered.

        Raises
        ------
        AssertionError
            If the objective isn't set due to the model being infeasible or unbounded.
        """
        
        assert self._objective is not None, 'model is infeasible or unbounded.'

        self._has_deliveries = True

        # defines the quality values of each parameter for each order
        self.__quality_mean()

        # iterates over a list of requests to save quality data
        requests: List[List[Request]] = [
            out.quality for out in self._problem.outputs
        ]

        # saves quality data for each parameter of each request
        for req, out in zip(requests, self._problem.outputs):
            quality_list: Qualities = [
                {
                    'parameter': quality.parameter,
                    'value': quality.value,
                    'minimum': quality.minimum, 
                    'maximum': quality.maximum,
                    'goal': quality.goal,
                    'importance': quality.importance
                } for quality in req
            ]

            # calculates the time the request was initiated and completed
            start: float
            end: float

            start, end = self.work_time(out.id)

            # calculates the optimal delivery duration
            optimal_duration: float = out.weight / sum([
                eng.speed_reclaim for eng in self.problem.engines
            ])

            # calculates the gap between the durations
            self._gap[out.id - 1] = round(
                1 - optimal_duration / (end - start),
                2
            )

            # add order information to the delivery list
            self._deliveries.append(
                {
                    'weight': out.weight,
                    'start_time': start,
                    'duration': round(end - start, 2),
                    'quality': quality_list
                }
            )

    def set_objective(self: 'Solution', objective: Objective) -> None:
        """
        Sets the objectives for the Machine Scheduling Problem from the results
        of the Ore Mixing Problem.

        Parameters
        ----------
        objective : Tuple[Optional[float], Dict[str, List[float]], Dict[str, List[float]]]
            A tuple where:
            - The first element is the objective value of the linear model.
            - The second element is a dictionary of entries with stockpile IDs as keys
              and lists of stacked weights as values.
            - The third element is a dictionary of recoveries with request IDs as keys
              and lists of reclaimed weights as values.
        """
        self._objective = objective[0]
        self._weights = objective[1]
        self._inputs = objective[2]

    def update_cost(self: 'Solution', _id: int) -> None:
        """Calculate and update the solution cost.

        Parameters
        ----------
        _id : int
            The request identifier.
        """
        self._cost = self.work_time(_id)[1]

    def work_time(self: 'Solution', _id: int) -> Tuple[float, float]:
        """
        Calculate and return the start and end times for a request.

        Parameters
        ----------
        _id : int
            The request identifier.

        Returns
        -------
        Tuple[float, float]
            A tuple where:
            - The first element is the start time.
            - The second element is the end time.

        Raises
        ------
        AssertionError
            If the reclaim list is empty when this method is called.
        """

        assert self._reclaims, 'calling work_time() for an empty reclaim list.'

        # calculates the time when the request was initiated
        start: float = min(
            [item['start_time']
             for item in self._reclaims if item['output'] == _id]
        )

        # calculates the time when the request was completed
        end: float = max(
            [item['start_time'] + item['duration']
             for item in self._reclaims if item['output'] == _id]
        )

        return start, end

    def write(self: 'Solution', file_path: str, time: float) -> None:
        """
        Write the solution to a JSON file.

        Parameters
        ----------
        file_path : str
            The output path.
        time : float
            The time when the solution was written.

        Raises
        ------
        AssertionError
            If this method is called before deliveries are set.
        """
        assert self._has_deliveries, 'calling write() before mandatory call to set_deliveries().'

        result: Result = {
            'info': self._problem.info,
            'objective': self._objective,
            'gap': self._gap,
            'stacks': self._stacks,
            'reclaims': self._reclaims,
            'outputs': self._deliveries,
            'time': time,
        }
        with open(file_path, 'w') as file:
            ujson.dump(result, file, indent=2)

    def reset(self: 'Solution') -> None:
        """Reset the solution to avoid the need to create another object."""
        self._stacks = []
        self._reclaims = []
        self._deliveries = []

    def __quality_mean(self: 'Solution') -> None:
        """
        Calculate and set the final quality values for each request.

        Raises
        ------
        AssertionError
            If the weight list is empty when this method is called.
        ZeroDivisionError
            If the model is infeasible or unbounded, which causes np.average() to raise an exception.
        """

        assert self._weights, 'calling __quality_mean() with a empty list of weights.'

        quality_list: List[List[float]] = [
            [quality.value for quality in stp.quality_ini]  
            for stp in self._problem.stockpiles
        ]

        try:
            # calculates the quality based on the weight taken from each pile
            mean: List[List[float]] = [
                list(np.average(quality_list, axis=0, weights=wl)) 
                for wl in list(self._weights.values())
            ]

            # assigns the calculated quality value to its respective parameter
            for quality, out in zip(mean, self._problem.outputs):
                for value, request in zip(quality, out.quality):
                    request.value = round(value, 2)

        # if the model is infeasible the np.average() function throws an exception
        except ZeroDivisionError:
            raise ZeroDivisionError('the model is infeasible or unbounded.')

    @property
    def problem(self: 'Solution') -> Problem:
        """Problem: The problem considered."""
        return self._problem

    @problem.setter
    def problem(self: 'Solution', value: Problem) -> None:
        self._problem = value
    
    @property
    def objective(self: 'Solution') -> Optional[float]:
        """The solution objective value."""
        return self._objective
    
    @objective.setter
    def objective(self: 'Solution', value: Optional[float]) -> None:
        self._objective = value

    @property
    def weights(self: 'Solution') -> Weights:
        """
        Dictionary of recoveries whose keys are the IDs of each request and the
        values are lists with the reclaimed weights.
        """
        return self._weights

    @weights.setter
    def weights(self: 'Solution', value: Weights) -> None:
        self._weights = value

    @property
    def inputs(self: 'Solution') -> Weights:
        """
        Dictionary of entries whose keys are the stockpiles IDs and values are
        lists with the stacked weights.
        """
        return self._inputs

    @inputs.setter
    def inputs(self: 'Solution', value: Weights) -> None:
        self._inputs = value

    @property
    def cost(self: 'Solution') -> float:
        """The solution cost."""
        # base_cost = self._cost
        # time_difference_penalty = self.calculate_time_difference_penalty()
        return self._cost

    @cost.setter
    def cost(self: 'Solution', value: float) -> None:
        self._cost = value

    @property
    def routes(self: 'Solution') -> Routes:
        """
        Matrix of routes tuples.

        The first element represents the stockpile position, and the last is the engine
        configuration in that stockpile. Each line indicates an engine, based on its IDs.
        """
        return self._routes

    @routes.setter
    def routes(self: 'Solution', value: Routes) -> None:
        self._routes = value

    @property
    def start_time(self: 'Solution') -> List[float]:
        """
        List with the time that each engine can start a new task.
        The indexes are associated with the IDs of each engine.
        """
        return self._start_time

    @start_time.setter
    def start_time(self: 'Solution', value: List[float]) -> None:
        self._start_time = value

    @property
    def gap(self: 'Solution') -> List[float]:
        """
        List with the gap between the optimal and current delivery duration.

        The indexes are associated with the IDs of each output.
        """
        return self._gap

    @gap.setter
    def gap(self: 'Solution', value: List[float]) -> None:
        self._gap = value

    @property
    def stacks(self: 'Solution') -> Jobs:
        """
        Dictionary with the stacking data to be recorded to a JSON file.
        The keys represent the names of the attributes and the values contain
        their information.
        """
        return self._stacks

    @stacks.setter
    def stacks(self: 'Solution', value: Jobs) -> None:
        self._stacks = value

    @property
    def reclaims(self: 'Solution') -> Jobs:
        """
        Dictionary with the reclaiming data to be recorded in a JSON file.

        The keys represent the names of each attribute, and the values contain
        their respective information.
        """
        return self._reclaims

    @reclaims.setter
    def reclaims(self: 'Solution', value: Jobs) -> None:
        self._reclaims = value

    @property
    def deliveries(self: 'Solution') -> Deliveries:
        """
        List of dictionaries with the final results, such as delivery time,
        ore weight, quality parameters, and other information for each request.
        """
        return self._deliveries

    @deliveries.setter
    def deliveries(self: 'Solution', value: Deliveries) -> None:
        self._deliveries = value

    @property
    def has_deliveries(self: 'Solution') -> bool:
        """
        Flag indicating whether deliveries are defined before writing them to a file.
        """
        return self._has_deliveries

    @has_deliveries.setter
    def has_deliveries(self: 'Solution', value: bool) -> None:
        self._has_deliveries = value
