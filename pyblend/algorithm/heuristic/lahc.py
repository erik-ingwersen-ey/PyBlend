import copy
from typing import List

from pyblend.algorithm.heuristic.heuristic import Heuristic
from pyblend.algorithm.neighborhood import Move, SmartSwap
from pyblend.model.problem import Problem
from pyblend.model.solution import Solution


class LAHC(Heuristic):
    """
    Class implements a Late Acceptance Hill-Climbing heuristic.
    """
    
    def __init__(
        self: 'LAHC', 
        problem: Problem,
        size: int,
    ):
        """
        Instantiate a new Late Acceptance Hill-Climbing.

        Parameters
        ----------
        problem : Problem
            The problem reference.
        size : int
            The number of most recent solutions.
        """

        super().__init__(problem, 'Late Acceptance Hill-Climbing')

        self.__size: int = size

    def run(
        self: 'LAHC', 
        initial_solution: Solution,
        max_iters: int,
        best_known: bool = False
    ) -> None:
        """
        Execute and update the Late Acceptance Hill-Climbing with the best solution.

        Parameters
        ----------
        initial_solution : Solution
            The initial input solution.
        max_iters : int
            The maximum number of iterations to execute.
        best_known : bool, default=False
            Set to `True` if the initial best_solution has already been established,
            and `False` otherwise.
            Note that the default value `False` results in the initial `best_solution`
            being defined as the `initial_solution`.
        """
        # list of costs for each solution
        cost_list: List[float] = [
            initial_solution.cost * 1.5 for _ in range(self.__size)
        ]

        if not best_known:
            self._best_solution = initial_solution

        solution: Solution = copy.deepcopy(initial_solution)

        # cost list index
        v: int = 0

        for _ in range(max_iters):
            move: Move = self.select_move(solution)
            move.do_move(solution)

            if (
                solution.cost <= move.initial_cost or
                solution.cost <= cost_list[v]
            ):
                self.accept_move(move)

                if solution.cost < self._best_solution.cost:
                    self._best_solution = copy.deepcopy(solution)

            else:
                self.reject_move(move)

            cost_list[v] = solution.cost
            v = (v + 1) % self.__size

    @property
    def size(self: 'LAHC') -> int:
        """The number of most recent solutions."""
        return self.__size

    @size.setter
    def size(self: 'LAHC', value: int) -> None:
        self.__size = value
