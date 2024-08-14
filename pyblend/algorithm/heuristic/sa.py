import copy
import math
import random

from pyblend.algorithm.heuristic.heuristic import Heuristic
from pyblend.algorithm.neighborhood import Move
from pyblend.model.problem import Problem
from pyblend.model.solution import Solution


class SA(Heuristic):
    """
    Class implements a Simulated Annealing heuristic.
    """

    def __init__(
        self: 'SA', 
        problem: Problem,
        alpha: float,
        t0: float,
        sa_max: int = int(1e3)
    ):
        """
        Instantiate a new Simulated Annealing.

        Parameters
        ----------
        problem : Problem
            The problem reference.
        alpha : float
            Cooling rate for the simulated annealing.
        t0 : float
            Initial temperature.
        sa_max : int, default=1000
            Number of iterations before updating the temperature.
        """

        super().__init__(problem, 'Simulated Annealing')

        self.__alpha: float = alpha
        self.__t0: float = t0
        self.__sa_max: int = sa_max
        
        self.__eps: float = 1e-6

    def run(
        self: 'SA', 
        initial_solution: Solution,
        max_iters: int,
        best_known: bool = False
    ) -> None:
        """
        Execute update the Simulated Annealing with the best solution.

        Parameters
        ----------
        initial_solution : Solution
            The initial input solution.
        max_iters : int
            The maximum number of iterations to execute.
        best_known : bool, default=False
            Set it to `True` if the initial `best_solution` has already
            been established, and `False` otherwise.
            Note that the default value `False` results in the initial `best_solution`
            being defined as the `initial_solution`.
        """

        if not best_known:
            self._best_solution = initial_solution

        solution: Solution = copy.deepcopy(initial_solution)
        temperature: float = self.__t0
        
        self._iters = 0
        while temperature > self.__eps and self._iters < max_iters:
            solution.start_time = initial_solution.start_time.copy()

            move: Move = self.select_move(solution)
            delta: float = move.do_move(solution)

            # if the solution improved
            if delta < 0:
                self.accept_move(move)
                self._iters = 0

                if solution.cost < self._best_solution.cost:
                    self._best_solution = copy.deepcopy(solution)

            # if the solution didn't improve, but is accepted
            elif delta == 0:
                self.accept_move(move)

            # The solution didn't improve but may be accepted with a probability.
            else:
                x: float = random.uniform(0, 1)
                if x < math.exp(-delta / temperature):
                    self.accept_move(move)

                # if the solution is rejected
                else:
                    self.reject_move(move)

            self._iters += 1
            temperature *= self.__alpha

            # if necessary, updates temperature
            if temperature < self.__eps:
                temperature = self.__t0

    @property
    def alpha(self: 'SA') -> float:
        """Cooling rate for the simulated annealing."""
        return self.__alpha

    @alpha.setter
    def alpha(self: 'SA', value: float) -> None:
        self.__alpha = value

    @property
    def t0(self: 'SA') -> float:
        """Initial temperature."""
        return self.__t0

    @t0.setter
    def t0(self: 'SA', value: float) -> None:
        self.__t0 = value

    @property
    def sa_max(self: 'SA') -> int:
        """
        The maximum number of iterations without improvements to execute.
        """
        return self.__sa_max

    @sa_max.setter
    def sa_max(self: 'SA', value: int) -> None:
        self.__sa_max = value

    @property
    def eps(self: 'SA') -> float:
        """
        Conceptual zero.

        This is mainly to avoid zero division errors.
        """
        return self.__eps

    @eps.setter
    def eps(self: 'SA', value: float) -> None:
        self.__eps = value
