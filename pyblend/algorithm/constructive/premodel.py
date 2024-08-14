from heapq import heapify, heappop
from typing import List

from pyblend.algorithm.constructive.constructive import Constructive
from pyblend.config import Route
from pyblend.model.classes import Engine
from pyblend.model.problem import Problem
from pyblend.model.solution import Solution


class PreModel(Constructive):
    """This class contains simple constructive procedures for the Machine
    Scheduling Problem. Generates and returns a greedy solution. The jobs are
    added in a specific order to the machine to which they incur the smallest
    increase in the makespan.
    """

    def __init__(self: "PreModel", problem: Problem, solution: Solution):
        """Instantiates a new PreModel Constructive for the Machine Scheduling
        Problem.

        Args:
            problem (Problem): The problem reference.
            solution (Solution): The OMP solution reference.
        """

        super().__init__(problem, solution)

        self._feed_back: List[List[float]] = [
            [1e3] * len(self._problem.stockpiles) for _ in self._problem.outputs
        ]

    def run(self: "PreModel", has_routes: bool = False) -> None:
        """Executes the Constructive for all output requests.

        Args:
            has_routes (bool): Flag to indicate if the routes are already
                defined. True if the routes have already been established,
                False otherwise. Note that the False option will define the
                routes automatically and greedily. Defaults to False.
        """

        super().run(has_routes)

    def build(self: "PreModel") -> None:
        """This method defines the operations performed on each stockpile and
        their durations.

        To use this method outside the run() method, you must manually assign
        the value of the output_id attribute of the Constructive class
        and the routes attribute of the Solution class before using it.
        """
        super().build()

    def set_routes(self: "PreModel") -> None:
        """This method defines the order of operation of all machines and save
        the result in the routes attribute of the Solution class.
        """

        super().set_routes()

    def set_route(self: "PreModel", start_time: List[float], engine: Engine) -> Route:
        """This method greedily defines the operating order of each
        machine. It assigns all possible jobs to each machine, which `set_jobs`
        must further refine().

        Args:
            start_time (List[float]): List with the time when each engine can
                start a new task.
            engine (Engine): The engine reference.

        Returns:
            List[Tuple[float, int, int, str]]: List of tuples whose first
                element is the access time of the machine to the stockpile,
                the second element is the engine ID, the third element is its
                position and the last element is its configuration.
        """

        # list to indicate whether the machine has already visited the stockpile
        visited: List[bool] = [
            False for stp in self._problem.stockpiles if engine.rail in stp.rails
        ]

        # list with machine routes and variable with its starting position
        route: Route = []
        pos: int = engine.pos_ini

        while not all(visited):
            try:
                faster: float
                pos: int

                # finds the stockpile with the shortest access time
                faster, pos = min(
                    (time_travel + start_time[engine.id - 1], i)
                    for i, (time_travel, is_visited) in enumerate(
                        zip(self._problem.time_travel[pos], visited)
                    )
                    if is_visited is False
                    and engine.rail in self._problem.stockpiles[i].rails
                )

                # indicates which activity will be performed by the machine
                # r to reclaim, s to stack, and b to both
                atv: str = "r"

                # updates the start time list
                start_time[engine.id - 1] += faster

                # adds data to the referenced engine's route list
                route.append((faster, engine.id, pos, atv))

                visited[pos] = True

            # if the model is infeasible, the min() function throws an exception
            except ValueError:
                break

        return route

    def set_jobs(self: "PreModel", routes: List[Route]) -> None:
        """This method defines greedily where each machine will act based on
        its routes and the start time of each job.
        """

        routes: Route = [item for sublist in routes for item in sublist]
        heapify(routes)

        while routes:
            route = heappop(routes)

            # route[_] is:
            # 0, the time taken to access the stockpile, used by the heap
            # 1, the machine ID, which will be its index in the list
            # 2, the position of the stockpile that will enter the machine's route
            # 3, the type of activity that will be performed on the stockpile

            eng: int
            stp: int
            atv: str

            eng, stp, atv = route[1] - 1, route[2], route[3]
            self._solution.routes[eng].append((stp, atv))

        for eng in self._solution.routes:
            for stp in eng:
                self._feed_back[self._output_id][stp[0]] = 1

    def reset_inputs(self: "PreModel") -> None:
        """This method is called whenever the input list should be reset
        (mainly to avoid the need of creating another object).
        """

        super().reset_inputs()
