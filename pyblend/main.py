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
    """Execute the main program logic for processing input and generating
    output.

    This function is responsible for parsing command-line arguments,
    initializing parameters, and orchestrating the solution process. It
    reads input data, constructs the necessary models and heuristics, and
    outputs the results to a specified file. The function also handles
    feedback mechanisms if specified in the parameters.
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
    """Execute the selected constructive method for the given problem.

    This function orchestrates the execution of either a pre-model or post-
    model constructive method based on the provided parameters. It first
    checks the 'constructive' parameter to determine which method to
    execute. If the method is 'premodel', it initializes the necessary
    inputs and outputs, sets the objective for the solution, and runs the
    pre-model process. The results are then used to update the linear model
    weights. If the method is 'postmodel', it resolves the model and runs
    the post-model process.

    Args:
        problem (Problem): The problem reference containing stockpiles and inputs.
        solution (Solution): The solution reference to set objectives.
        model (LinModel): The linear model to be updated with weights.
        parms (Parameters): The operating guidelines that dictate the method to use.

    Returns:
        Constructive: The constructive procedure that was executed.
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
    """Run the selected heuristic approach to solve a given problem.

    This function initializes and executes a heuristic solver based on the
    specified algorithm in the parameters. It supports different algorithms
    such as LAHC (Late Acceptance Hill Climbing) and SA (Simulated
    Annealing). The function creates neighborhoods for the problem and runs
    the solver with the provided solution and operational guidelines.

    Args:
        problem (Problem): The problem reference to be solved.
        solution (Solution): The solution reference where results will be stored.
        constructive (Constructive): The constructive procedure used in the solving process.
        parms (Parameters): The operating guidelines that dictate the algorithm and its parameters.

    Returns:
        Optional[Heuristic]: The heuristic procedure used for solving the problem, or None if no
            valid algorithm was selected.
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
    """Create neighborhoods for the heuristic.

    This function initializes various neighborhood structures for a given
    heuristic solver. It adds multiple move strategies to the solver, which
    are essential for exploring the solution space of the problem. The moves
    include different types of shifts and swaps that can be utilized during
    the heuristic search process.

    Args:
        problem (Problem): The problem reference that defines the context
            in which the neighborhoods are created.
        solver (Heuristic): The heuristic procedure that will utilize the
            neighborhoods for solving the problem.
        constructive (Constructive): The constructive procedure that aids
            in generating solutions within the defined neighborhoods.

    Returns:
        None: This function does not return any value; it modifies the
            solver in place by adding move strategies.
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
    """Run the feedback approach to optimize a solution using a linear model.

    This function iteratively applies a feedback mechanism to improve the
    given solution based on the specified linear model and constructive
    procedures. It updates the model with the current weights from the
    solution, resolves the model to obtain an objective, and then runs the
    constructive procedure. If a solver is provided, it will also execute
    the solver on the updated solution for a specified number of iterations.

    Args:
        solution (Solution): The solution reference to be optimized.
        model (LinModel): The linear model used for optimization.
        solver (Optional[Heuristic]): The heuristic procedure for further
            optimization, if provided.
        constructive (Constructive): The constructive procedure to apply
            during the feedback loop.
        parms (Parameters): The operating guidelines, including feedback
            iterations and maximum iterations for the solver.
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

    This function processes command-line arguments and updates the provided
    parameters dictionary based on the options specified. It expects a list
    of arguments that may include various flags and their corresponding
    values. If the number of arguments is less than three, it will print
    usage information. The function supports multiple options, including
    settings for constructive methods, algorithms, feedback levels, random
    seed values, maximum iterations, and parameters specific to local search
    and simulated annealing.

    Args:
        args (List[str]): The terminal argument list.
        parms (Parameters): The operating guidelines.
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

    This function prints the usage instructions for the program, detailing
    the required input and output files, as well as the available options
    and parameters that can be specified by the user. It provides a clear
    guide on how to run the program, including examples of command-line
    usage.

    Args:
        parms (Parameters): The operating guidelines containing default
            values for various options.

    Returns:
        None: This function does not return a value; it prints the usage
            information to the console and exits the program.
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
