import os
import random
from typing import Dict, Optional, Tuple

from mip import LinExpr, Model, Var, xsum

from pyblend.config import Inputs, Objective, Outputs, Stockpiles
from pyblend.model.problem import Problem


class LinModel:
    """
    Linear Model built using the Python-MIP package.
    """

    def __init__(self: "LinModel", problem: Problem):
        """
        Instantiate the `LinModel` class.

        Parameters
        ----------
        problem : Problem
            Problem considered.
        """

        self._omp: Model = Model("Ore Mixing Problem", solver_name="cbc")

        # Problem data used to solve the model
        self._info: str = problem.info[0]
        self._stockpiles: Stockpiles = problem.stockpiles
        self._outputs: Outputs = problem.outputs
        self._inputs: Inputs = problem.inputs

        # set of stockpiles, quality parameters, requests, and ore inputs
        self._p: int = len(problem.stockpiles)
        self._t: int = len(problem.outputs[0].quality)
        self._r: int = len(problem.outputs)
        self._e: int = len(problem.inputs)

        # variables for the Ore Mixing Problem
        self._x: Optional[Var] = None
        self._y: Optional[Var] = None

        # weights of the restrictions in the objective function
        self._w_1: int = problem.info[1]
        self._w_2: int = problem.info[2]

        # variable weights for the Ore Mixing Problem
        self._w_x: Dict[Tuple[int, int], int] = {
            (i, k): 1 for i in range(self._p) for k in range(self._r)
        }

        self._w_y: Dict[Tuple[int, int], int] = {
            (h, i): 1 for h in range(self._e) for i in range(self._p)
        }

        # deviation variables for the Ore Mixing Problem
        self._a_max: Optional[Var] = None
        self._a_min: Optional[Var] = None
        self._b_max: Optional[Var] = None
        self._b_min: Optional[Var] = None

        # control flags
        self.__has_vars: bool = False
        self.__has_constrs: bool = False
        self.__has_objective: bool = False

        # assigns variable values, creates constraints and objective function
        self.__add_vars()
        self.__add_constrs()
        self.__add_objective()

    def resolve(self: "LinModel") -> Objective:
        """
        Solve the linear model and write the problem details to an LP file.

        Returns
        -------
        Tuple[Optional[float], Dict[str, List[float]], Dict[str, List[float]]]
            Returns a tuple. The first element is the objective value of the
            model. The second element is a dictionary with stockpiles IDs as keys
            and lists of stacked weights as values. The last element represents
            a dictionary of recoveries with request IDs as keys and reclaimed weights
            as values.
        """

        assert (
            self.__has_objective
        ), "calling the resolve() before mandatory call to __add_objective()."

        # solving the model
        os.makedirs(os.path.dirname(f"./out/logs/"), exist_ok=True)
        self._omp.write(f"./out/logs/{self._info}.lp")
        self._omp.optimize()

        if self._omp.num_solutions > 0:
            # output weights taken from each stockpile i for each request k
            reclaims = {
                f"id: {self._outputs[k].id}": [self._x[i, k].x for i in range(self._p)]
                for k in range(self._r)
            }

            # input weights taken from each input j for each stockpile k
            inputs = {
                f"id: {self._stockpiles[i].id}": [
                    self._y[h, i].x for h in range(self._e)
                ]
                for i in range(self._p)
            }
            return self._omp.objective_value, reclaims, inputs

        return None, {}, {}

    def add_weights(
        self: "LinModel", variable: str, weights: Dict[Tuple[int, int], int]
    ) -> None:
        """
        Assign weights to the variables.

        Method should be called whenever it's necessary to send feedback to the model.

        Parameters
        ----------
        variable : str
            Indicator of which variable weights are defined.
            It must be 'x' or 'y'.
        weights : Dict[Tuple[int, int], int]
            A dictionary of reclaimed or stacked weights from a previous model
            execution.
        """
        assert variable == "x" or variable == "y", (
            "The variable 'x' or 'y' to which the weights "
            "will be applied must be defined."
        )

        assert weights, "Calling add_weights with an empty matrix of weights."

        if variable == "x":
            # resets the previous list of weights, if any
            self._w_x = {(i, k): 1 for i in range(self._p) for k in range(self._r)}

            # sets a new list of weights with random values
            for k, lin in enumerate(weights):
                for i, col in enumerate(lin):
                    self._w_x[i, k] = random.randint(1, 1000) if col > 0 else 1

        elif variable == "y":
            # resets the previous list of weights, if any
            self._w_y = {(h, i): 1 for h in range(self._e) for i in range(self._p)}

            # sets a new list of weights with random values
            for i, lin in enumerate(weights):
                for h, col in enumerate(lin):
                    self._w_y[h, i] = random.randint(1, 1000) if col > 0 else 1

    def __add_vars(self: "LinModel") -> None:
        """
        Assign values to variables.

        This method is called automatically during the class instantiation,
        and there's no need to use it afterward.
        """

        assert not self.__has_vars, (
            "calling the __add_vars() private method that was already "
            "executed when instantiating the class."
        )

        self.__has_vars = True

        # x_ik is the quantity of ore removed from stockpile i for request k
        self._x = {
            (i, k): self._omp.add_var(name=f"x_{i}{k}")
            for i in range(self._p)
            for k in range(self._r)
        }

        # y_hk is the quantity of ore removed from input h for stockpile i
        self._y = {
            (h, i): self._omp.add_var(name=f"y_{h}{i}")
            for h in range(self._e)
            for i in range(self._p)
        }

        # var_jk is the deviation from the quality parameter j of the request k
        self._a_max = {
            (j, k): self._omp.add_var(name=f"a_max_{j}{k}")
            for j in range(self._t)
            for k in range(self._r)
        }
        self._a_min = {
            (j, k): self._omp.add_var(name=f"a_min_{j}{k}")
            for j in range(self._t)
            for k in range(self._r)
        }
        self._b_max = {
            (j, k): self._omp.add_var(name=f"b_max_{j}{k}")
            for j in range(self._t)
            for k in range(self._r)
        }
        self._b_min = {
            (j, k): self._omp.add_var(name=f"b_min_{j}{k}")
            for j in range(self._t)
            for k in range(self._r)
        }

    def __add_constrs(self: "LinModel") -> None:
        """
        Create constraints for the model.

        This method is called automatically during the class instantiation,
        and there's no need to use it afterward.
        """

        assert (
            self.__has_vars
        ), "calling the __add_constrs() before mandatory call to __add_vars()."

        assert not self.__has_constrs, (
            "calling the __add_constrs() private method that was already "
            "executed when instantiating the class."
        )

        self.__has_constrs = True

        # capacity constraint of inputs
        for h in range(self._e):
            self._omp += (
                xsum(self._y[h, i] for i in range(self._p)) <= self._inputs[h].weight,
                f"input_weight_constr_{h}",
            )

        # stockpile capacity constraints
        for i in range(self._p):
            self._omp += (
                xsum(self._y[h, i] for h in range(self._e))
                + self._stockpiles[i].weight_ini
                <= self._stockpiles[i].capacity,
                f"capacity_constr_{i}",
            )

            for h in range(self._e):
                self._omp += (
                    xsum(self._x[i, k] for k in range(self._r))
                    <= self._stockpiles[i].weight_ini + self._y[h, i],
                    f"weight_constr_{i}{h}",
                )

        for k in range(self._r):
            # demand constraint
            self._omp += (
                xsum(self._x[i, k] for i in range(self._p)) == self._outputs[k].weight,
                f"demand_constr_{k}",
            )

            # quality constraints
            for j in range(self._t):
                # minimum quality deviation constraint
                q_1: LinExpr = xsum(
                    self._x[i, k]
                    * (
                        self._stockpiles[i].quality_ini[j].value
                        - self._outputs[k].quality[j].minimum
                    )
                    for i in range(self._p)
                )

                self._omp += (
                    q_1 + self._a_min[j, k] * self._outputs[k].weight >= 0,
                    f"min_quality_constr_{j}{k}",
                )

                # maximum quality deviation constraint
                q_2: LinExpr = xsum(
                    self._x[i, k]
                    * (
                        self._stockpiles[i].quality_ini[j].value
                        - self._outputs[k].quality[j].maximum
                    )
                    for i in range(self._p)
                )

                self._omp += (
                    q_2 - self._a_max[j, k] * self._outputs[k].weight <= 0,
                    f"max_quality_constr_{j}{k}",
                )

                # deviation constraint from the quality goal
                q_3: LinExpr = xsum(
                    self._x[i, k]
                    * (
                        self._stockpiles[i].quality_ini[j].value
                        - self._outputs[k].quality[j].goal
                    )
                    for i in range(self._p)
                )

                self._omp += (
                    q_3
                    + (self._b_min[j, k] - self._b_max[j, k]) * self._outputs[k].weight
                    == 0,
                    f"goal_quality_constr_{j}{k}",
                )

    def __add_objective(self: "LinModel") -> None:
        """
        Create the objective function for the model.

        This method is called automatically during the class instantiation,
        and there's no need to use it afterward.
        """

        assert self.__has_constrs, (
            "calling the __add_objective() before mandatory " "call to __add_constrs()."
        )

        assert not self.__has_objective, (
            "calling the __add_objective() private method that was "
            "already executed when instantiating the class."
        )

        self.__has_objective = True

        # deviation from limits
        d_limit: LinExpr = xsum(
            self._outputs[k].quality[j].importance
            * self._a_min[j, k]
            / self.__normalize(j, k, "lb")
            + self._outputs[k].quality[j].importance
            * self._a_max[j, k]
            / self.__normalize(j, k, "ub")
            for j in range(self._t)
            for k in range(self._r)
        )

        # goal deviation
        d_goal: LinExpr = xsum(
            (self._b_min[j, k] + self._b_max[j, k])
            / min(self.__normalize(j, k, "lb"), self.__normalize(j, k, "ub"))
            for j in range(self._t)
            for k in range(self._r)
        )

        # scheduling reclaims
        r_scheduling: LinExpr = xsum(
            self._w_x[i, k] * self._x[i, k]
            for i in range(self._p)
            for k in range(self._r)
        )

        # scheduling inputs
        i_scheduling: LinExpr = xsum(
            self._w_y[h, i] * self._y[h, i]
            for h in range(self._e)
            for i in range(self._p)
        )

        # Minimize the total scheduling time
        total_time: LinExpr = xsum(
            self._x[i, k] * self._w_x[i, k] for i in range(self._p) for k in
            range(self._r)
        )

        # objective function
        self._omp += (
            self._w_1 * d_limit + self._w_2 * d_goal + r_scheduling + i_scheduling + total_time * 1_000
        )

    def __normalize(self: "LinModel", j: int, k: int, bound: str) -> float:
        """
        Calculate the units of deviation and avoids division by zero.

        This method is called by `__add_objective` method, and there's no need
        to use it afterward.

        Parameters
        ----------
        j : int
            The considered quality parameter index.
        k : int
            The considered request index.
        bound : str
            Indicator if the calculation should be made by the
            upper bound or the lower bound. This argument must be defined
            as 'ub' for the upper bound or 'lb' for the lower bound.

        Returns
        -------
        float
            The result of subtracting the quality goal by the
            indicated limit. If this calculation is equal to zero,
            then the value of 1e-6 is returned.
        """
        assert (
            bound == "ub" or bound == "lb"
        ), "the upper or lower bound indicator must be defined."

        # outputs[k].quality[j] is the quality parameter j of the request k
        ans: float = 0

        # ub indicates that the maximum should normalize the calculation
        if bound == "ub":
            ans = self._outputs[k].quality[j].maximum - self._outputs[k].quality[j].goal

        # lb indicates that the minimum should normalize the calculation
        elif bound == "lb":
            ans = self._outputs[k].quality[j].goal - self._outputs[k].quality[j].minimum

        return ans if ans != 0 else 1e-6
