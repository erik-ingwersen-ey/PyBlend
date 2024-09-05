import random

from typing import List, Optional

from pyblend.algorithm.neighborhood import Move, SmartSwap
from pyblend.model.problem import Problem
from pyblend.model.solution import Solution


class Heuristic:
    """
    This class represents a Heuristic, also known as, Local Search method.

    The basic methods and neighborhood selection are included.
    """

    def __init__(
        self: 'Heuristic', 
        problem: Problem,
        name: str,
    ):
        """
        Instantiate a new Heuristic.

        Parameters
        ----------
        problem:  Problem
            The problem reference.
        name : str
            The name of the heuristic.
        """

        self._problem: Problem = problem
        self._name: str = name

        self._moves: List[Move] = []
        self._best_solution: Optional[Solution] = None
        self._iters: int = 0

    def add_move(self: 'Heuristic', move: Move) -> None:
        """
        Add a move to the heuristic.
        
        Parameters
        ----------
        move : int
            The move to add.
        """
        self._moves.append(move)
    
    def accept_move(self: 'Heuristic', move: Move) -> None:  # noqa
        """
        Accept a move.

        Parameters
        ----------
        move : Move
            The move to accept.
        """
        move.accept()

    def reject_move(self: 'Heuristic', move: Move) -> None:  # noqa
        """
        Reject a move.

        Parameters
        ----------
        move : Move
            The move to reject.
        """
        move.reject()

    def select_move(self: 'Heuristic', solution: Solution) -> Move:
        """
        Select a move.

        Parameters
        ----------
        solution : Solution
            The solution.

        Returns
        -------
        Move
            A randomly selected move/neighborhood.
        """
        size: int = len(self._moves)

        move: Move = self._moves[random.randrange(0, size)]
        move.gen_move(solution)

        while not move.has_move(solution):
            move = self._moves[(random.randrange(0, size))]
            move.gen_move(solution)

        return move

    @property
    def problem(self: 'Heuristic') -> Problem:
        """The problem reference."""
        return self._problem

    @problem.setter
    def problem(self: 'Heuristic', value: Problem) -> None:
        self._problem = value

    @property
    def name(self: 'Heuristic') -> str:
        """The name of the heuristic."""
        return self._name

    @name.setter
    def name(self: 'Heuristic', value: str) -> None:
        self._name = value

    @property
    def moves(self: 'Heuristic') -> List[Move]:
        """List with all the movements present in this heuristic."""
        return self._moves

    @moves.setter
    def moves(self: 'Heuristic', value: List[Move]) -> None:
        self._moves = value
    
    @property
    def best_solution(self: 'Heuristic') -> Optional[Solution]:
        """Best solution found by the heuristic."""
        return self._best_solution

    @best_solution.setter
    def best_solution(self: 'Heuristic', value: Optional[Solution]) -> None:
        self._best_solution = value

    @property
    def iters(self: 'Heuristic') -> int:
        """Iteration counter."""
        return self._iters

    @iters.setter
    def iters(self: 'Heuristic', value: int) -> None:
        self._iters = value
