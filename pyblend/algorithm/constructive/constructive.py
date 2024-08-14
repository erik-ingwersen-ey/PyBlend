from typing import List, Optional

from pyblend.config import Route
from pyblend.model.classes import Engine
from pyblend.model.problem import Problem
from pyblend.model.solution import Solution


class Constructive:
    """
    Simple constructive procedures for the Machine Scheduling Problem.

    Generates and returns a greedy solution. The jobs are added in a specific
    order to the machine to which they generate the smallest increase in the
    'makespan' possible.

    """

    def __init__(self: "Constructive", problem: Problem, solution: Solution):
        """
        Instantiate the `Constructive` class for the Machine Scheduling Problem.

        Parameters
        ----------
        problem : Problem
            The problem reference.
        solution : Solution
            The OMP solution reference.
        """

        self._problem: Problem = problem
        self._solution: Solution = solution

        self._output_id: Optional[int] = None
        self._weights: List[List[float]] = list(solution.weights.values())
        self._inputs: List[float] = [sum(inp) for inp in list(solution.inputs.values())]

    def run(self: "Constructive", has_routes: bool = False) -> None:
        """
        Execute the `Constructive` for all output requests.

        Parameters
        ----------
        has_routes : bool, default=False
            Flag that indicates if the routes have been already defined.
            Set it to `True` if the routes have already been established,
            and `False` otherwise. Note that the default value `False` results
            in the routes being automatically defined greedily.
        """

        # Resets the solution start time for each execution
        self._solution._start_time = [0] * len(self._problem.engines)

        # The output_id must have already been specified for the defined route
        if has_routes:
            self.build()

        else:
            for out in self._problem.outputs:
                self._output_id = out.id - 1
                self.set_routes()
                self.build()

        self.reset_inputs()

    def build(self: "Constructive") -> None:
        """
        Define the operations performed on each stockpile and their durations.

        Notes
        -----
        To use this method outside the `run` method, you have to manually assign
        the value for the 'output_id' attribute of the `Constructive` class
        and the routes attribute of the `Solution` class prior to using it.
        """
        assert self._solution.weights, "calling build() with an empty list of weights."
        assert self._solution.inputs, "calling build() with an empty list of inputs."
        assert self._output_id >= 0, "calling build() before specifying the output request ID."
        assert self._solution.routes != [], "calling build() before defining the routes for each machine."

        # Reset the solution to save new results
        self._solution.reset()

        for eng, route in zip(self._problem.engines, self._solution.routes):
            for stp, atv in route:
                # Setup time, if there's more than one job in the same stockpile
                setup_time: float = 0.0

                # Reclaimer time
                duration: float = (
                    round(self._weights[self._output_id][stp] / eng.speed_reclaim, 2)
                    if eng.speed_reclaim > 0
                    else 0
                )

                # Travel time and setup to stockpile
                time_travel: float = self._problem.time_travel[eng.pos_ini][stp]

                # Performs the stacking activity before performing the reclaiming
                if atv == "s" or atv == "b":
                    self._solution.stacks.append(
                        {
                            "weight": round(self._inputs[stp], 1),
                            "stockpile": stp + 1,
                            "engine": eng.id,
                            "start_time": round(
                                self._solution.start_time[eng.id - 1] + time_travel, 2
                            ),
                            "duration": round(self._inputs[stp] / eng.speed_stack, 2),
                        }
                    )

                    # Adds stacking time if there's any input
                    self._solution.start_time[eng.id - 1] += self._solution.stacks[-1][
                        "duration"
                    ]
                    setup_time += self._problem.time_travel[stp][stp]
                    self._inputs[stp] = 0.0

                # Ore reclaim activity from the stockpile
                if atv == "r" or atv == "b":
                    self._solution.reclaims.append(
                        {
                            "weight": round(self._weights[self._output_id][stp], 1),
                            "stockpile": stp + 1,
                            "engine": eng.id,
                            "start_time": round(
                                self._solution.start_time[eng.id - 1]
                                + time_travel
                                + setup_time,
                                2,
                            ),
                            "duration": duration,
                            "output": self._output_id + 1,
                        }
                    )

                self._solution.start_time[eng.id - 1] += duration + time_travel

            # Changes the starting position of the machine
            try:
                eng.pos_ini = route[-1][0]

            # If the machine hasn't received any jobs
            except IndexError:
                pass

        # Updates the cost
        self._solution.update_cost(self._output_id + 1)

    def set_routes(self: "Constructive") -> None:
        """
        Define the order of operation of all machines.

        This method then saves the results in the routes attribute of the
        `Solution` class instance.
        """

        routes: List[Route] = []
        start_time: List[float] = self._solution.start_time.copy()

        # Appends the individual machine's route to the route list
        for engine in self._problem.engines:
            routes.append(self.set_route(start_time, engine))

        self.set_jobs(routes)

    def set_route(
        self: "Constructive", start_time: List[float], engine: Engine
    ) -> Route:
        """
        Define the operating order of each machine greedily.

        It assigns all possible jobs to each machine, which must be further
        refined by the `set_jobs` method.

        Parameters
        ----------
        start_time : List[float]
            List with the time when each engine can
                start a new task.
        engine : Engine
            The engine reference.

        Returns
        -------
        List[Tuple[float, int, int, str]]
            Returns the list of tuples where the first element is the access time
            of each machine to a stockpile. the second and third elements are
            the engine ID and its position. The final element contains the engine
            configurations.
        """
        raise NotImplementedError

    def set_jobs(self: "Constructive", routes: List[Route]) -> None:
        """
        Define jobs greedily.

        Each machine acts based on its routes and the start time of each job.
        """
        raise NotImplementedError

    def reset_inputs(self: "Constructive") -> None:
        """
        Reset the input list.

        This is mainly used to avoid the need of creating new objects.
        """

        self._inputs = [sum(inp) for inp in list(self._solution.inputs.values())]

    @property
    def problem(self: "Constructive") -> Problem:
        """The considered `Problem` instance."""
        return self._problem

    @problem.setter
    def problem(self: "Constructive", value: Problem) -> None:
        self._problem = value

    @property
    def solution(self: "Constructive") -> Solution:
        """The solution reference."""
        return self._solution

    @solution.setter
    def solution(self: "Constructive", value: Solution) -> None:
        self._solution = value

    @property
    def output_id(self: "Constructive") -> Optional[int]:
        """The output request identifier."""
        return self._output_id

    @output_id.setter
    def output_id(self: "Constructive", value: Optional[int]) -> None:
        self._output_id = value

    @property
    def weights(self: "Constructive") -> List[List[float]]:
        """
        List with the weights retrieved from each output request.

        The lines represent the requests and the stockpiles its columns.
        """
        return self._weights

    @weights.setter
    def weights(self: "Constructive", value: List[List[float]]) -> None:
        self._weights = value

    @property
    def inputs(self: "Constructive") -> List[float]:
        """
        List with stacked weights of the stockpiles.
        """
        return self._inputs

    @inputs.setter
    def inputs(self: "Constructive", value: List[float]) -> None:
        self._inputs = value
