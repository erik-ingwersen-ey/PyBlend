import random
import sys
import time
from pathlib import Path
from typing import List, Optional

from pyblend.algorithm.constructive import Constructive, LinModel, PostModel, PreModel
from pyblend.algorithm.heuristic import Heuristic, LAHC, SA
from pyblend.algorithm.neighborhood import (Shift, SimpleSwap, SmartShift,
                                            SmartSimpleSwap, SmartSwap, SmartSwitch,
                                            Swap, Switch)
from pyblend.config import Objective, Parameters
from pyblend.model.problem import Problem
from pyblend.model.solution import Solution
from pyblend.utils import handle_input_path, handle_output_path


def main():
    """
    This is the main function, responsible for parsing the input,
    instantiating moves and heuristics, and printing the results.
    """
    start_time = time.time()
    parms: Parameters = {
        "constructive": "postmodel",
        "algorithm": "",
        "feedback": 0,
        "seed": 0,
        "maxiters": int(1e3),
        "lsize": int(1e3),
        "alpha": 0.9,
        "samax": int(1e3),
        "t0": 1.0,
    }

    read_args(sys.argv, parms)

    random.seed(parms["seed"])

    input_path = str(Path(sys.argv[1]).resolve())
    output_path = handle_output_path(sys.argv[2])

    problem: Problem = Problem(input_path)
    solution: Solution = Solution(problem)
    model: LinModel = LinModel(problem)

    constructive: Constructive = construct(problem, solution, model, parms)

    solver: Optional[Heuristic] = None
    if parms["algorithm"] != "":
        solver = solve(problem, solution, constructive, parms)
        solution = solver.best_solution

    if parms["feedback"] > 0:
        feedback_approach(solution, model, solver, constructive, parms)

    solution.set_deliveries()
    solution.write(output_path, round(time.time()-start_time, 2))


def construct(
    problem: Problem, solution: Solution, model: LinModel, parms: Parameters
) -> Constructive:
    """This function executes the selected constructive method.

    Parameters
    ----------
    problem : Problem
        The problem reference.
    solution : Solution
        The solution reference.
    model : LinModel
        The linear model.
    parms : Parameters
        The operating guidelines.

    Returns
    -------
    Constructive
        The constructive procedure.
    """
    constructive: Constructive

    if parms["constructive"] != "premodel" and parms["constructive"] != "postmodel":
        print_usage(parms)

    elif parms["constructive"] == "premodel":
        reclaims = {
            f"id: {k.id}": [i.weight_ini for i in problem.stockpiles]
            for k in problem.outputs
        }

        inputs = {
            f"id: {k.id}": [i.weight for i in problem.inputs]
            for k in problem.stockpiles
        }

        objective: Objective = (None, reclaims, inputs)
        solution.set_objective(objective)

        constructive = PreModel(problem, solution)
        constructive.run()

        model.add_weights("x", list(constructive._feed_back))
        model.add_weights("y", list(constructive._feed_back))

    objective: Objective = model.resolve()
    solution.set_objective(objective)

    constructive = PostModel(problem, solution)
    constructive.run()

    return constructive


def solve(
    problem: Problem, solution: Solution, constructive: Constructive, parms: Parameters
) -> Optional[Heuristic]:
    """This functions runs the selected heuristic approach.

    Parameters
    ----------
    problem : Problem
        The problem reference.
    solution : Solution
        The solution reference.
    constructive : Constructive
        The constructive procedure.
    parms : Parameters
        The operating guidelines.

    Returns
    -------
    Optional[Heuristic]
        The heuristic procedure.
    """
    solver: Optional[Heuristic] = None
    if parms["algorithm"] == "lahc":
        solver = LAHC(problem, parms["lsize"])
    elif parms["algorithm"] == "sa":
        solver = SA(problem, parms["alpha"], parms["t0"], parms["samax"])
    else:
        print_usage(parms)

    create_neighborhoods(problem, solver, constructive)
    solver.run(solution, parms["maxiters"])

    return solver


def create_neighborhoods(
    problem: Problem, solver: Heuristic, constructive: Constructive
) -> None:
    """This function creates the neighborhoods for the heuristic.

    Parameters
    ----------
    problem : Problem
        The problem reference.
    solver : Heuristic
        The heuristic procedure.
    constructive : Constructive
        The constructive procedure.
    """
    solver.add_move(Shift(problem, constructive))
    solver.add_move(SimpleSwap(problem, constructive))
    solver.add_move(Swap(problem, constructive))
    solver.add_move(Switch(problem, constructive))
    solver.add_move(SmartShift(problem, constructive))
    solver.add_move(SmartSimpleSwap(problem, constructive))
    solver.add_move(SmartSwap(problem, constructive))
    solver.add_move(SmartSwitch(problem, constructive))


def feedback_approach(
    solution: Solution,
    model: LinModel,
    solver: Optional[Heuristic],
    constructive: Constructive,
    parms: Parameters,
) -> None:
    """Run the feedback approach.

    Parameters
    ----------
    solution : Solution
        The solution reference.
    model : LinModel
        The linear model.
    solver : Heuristic
        The heuristic procedure.
    constructive : Constructive
        The constructive procedure.
    parms : Parameters
        The operating guidelines.
    """
    for _ in range(parms["feedback"]):
        model.add_weights("x", list(solution.weights.values()))
        model.add_weights("y", list(solution.inputs.values()))

        objective: Objective = model.resolve()
        solution.set_objective(objective)

        constructive.run()
        if solver is not None:
            solver.run(solution, parms["maxiters"], True)


def read_args(args: List[str], parms: Parameters) -> None:
    """Read the input arguments.

    Parameters
    ----------
    args : Lst[str]
        The terminal argument list.
    parms : Parameters
        The operating guidelines.
    """
    if len(args) < 3:
        print_usage(parms)

    index: int = 3
    while index < len(args):
        option: str = args[index]
        index += 1

        if option == "-constructive":
            parms["constructive"] = args[index]
        elif option == "-algorithm":
            parms["algorithm"] = args[index]
        elif option == "-feedback":
            parms["feedback"] = int(args[index])
        elif option == "-seed":
            parms["seed"] = int(args[index])
        elif option == "-maxiters":
            parms["maxiters"] = int(args[index])

        # LAHC
        elif option == "-lsize":
            parms["lsize"] = int(args[index])

        # SA
        elif option == "-alpha":
            parms["alpha"] = float(args[index])
        elif option == "-samax":
            parms["samax"] = int(args[index])
        elif option == "-t0":
            parms["t0"] = float(args[index])
        else:
            print_usage(parms)
        index += 1


def print_usage(parms: Parameters) -> None:
    """Print the program usage.

    Parameters
    ----------
    parms : Parameters
        The operating guidelines.
    """
    usage: str = (
        f"Usage: python pyblend <input> <output> [options]\n"
        + f"    <input>  : Name of the problem input file.\n"
        + f"    <output> : Name of the (output) solution file.\n"
        + f"\nOptions:\n"
        + f'    -constructive <constructive> : premodel, postmodel (default: {parms["constructive"]}).\n'
        + f"    -algorithm <algorithm>       : lahc, sa.\n"
        + f'    -feedback <feedback>         : maximum number of feedback interactions with the model (default: {parms["feedback"]}).\n'
        + f'    -seed <seed>                 : random seed (default: {parms["seed"]}).\n'
        + f'    -maxiters <maxiters>         : maximum number of interactions (default: {parms["maxiters"]}).\n'
        + f"\n    LAHC parameters:\n"
        + f'        -lsize <lsize> : LAHC list size (default: {parms["lsize"]}).\n'
        + f"\n    SA parameters:\n"
        + f'        -alpha <alpha> : cooling rate for the Simulated Annealing (default: {parms["alpha"]}).\n'
        + f'        -samax <samax> : iterations before updating the temperature for Simulated Annealing (default: {parms["samax"]}).\n'
        + f'        -t0 <t0>       : initial temperature for the Simulated Annealing (default: {parms["t0"]}). \n'
        + f"\nExamples:\n"
        + f"    python pyblend instance_1.json out_1.json\n"
        + f"    python pyblend instance_1.json out_1.json -constructive premodel -seed 1\n"
        + f"    python pyblend instance_1.json out_1.json -algorithm sa -alpha 0.98 -samax 1000 -t0 1e5\n"
    )

    print(usage)
    sys.exit()


if __name__ == "__main__":
    main()
