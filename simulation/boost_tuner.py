from jmetal.algorithm.multiobjective.spea2 import SPEA2
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.termination_criterion import StoppingByTime

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.util.solution import get_non_dominated_solutions
from jmetal.util.evaluator import MultiprocessEvaluator, RayEvaluator


import numpy as np
import os, sys, argparse
import csv
import copy, bisect
import matplotlib.pyplot as plt
import pickle
from fractions import Fraction as frac
from datetime import datetime, timedelta
from functools import partial


from models import Models
from PriorityScheduler import AppPrioScheduler
from common import App, Job, Event
from sim import *


import pickle
import argparse
import ray


class Objective(object):
    """docstring for Objective"""

    def __init__(self, lmda, label):
        super(Objective, self).__init__()
        self._lmda = lmda
        self._label = label

    def __call__(self, entry):
        return self._lmda(entry)

    def get_name(self):
        return self._label


class BoostTune(FloatProblem):
    """docstring for BoostTune"""

    def __init__(self, total_gpus, app_list, event_queue, objectives):
        super(BoostTune, self).__init__()

        self._total_gpus = total_gpus
        self._app_list = app_list
        self._event_queue = event_queue

        self._service_times = list()
        for app_id in app_list:
            self._service_times.append(app_list[app_id].service)
        self._service_times.sort()

        self.objectives = objectives[:]
        self.number_of_objectives = len(objectives)
        self.obj_directions = [self.MINIMIZE] * self.number_of_objectives
        self.obj_labels = [objective.get_name() for objective in objectives]

        self.number_of_variables = 1
        self.number_of_constraints = 0

        # boost gamma
        self.lower_bound = [1e-11]
        
        # fix this later
        self.upper_bound = [5]

        # ['mean_pred','mean_jct']

    def get_bounds(self):
        return [self.lower_bound, self.upper_bound]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:



        tick = datetime.now()
        scheduler = AppPrioScheduler(
            total_gpus=self._total_gpus,
            event_queue=copy.deepcopy(self._event_queue),
            app_list=copy.deepcopy(self._app_list),
            prio_func=lambda a: (tick - a.submit_time).total_seconds() - ((1.0/solution) * math.log(1.0/(1.0-math.exp(-1.0*solution*a.estimated_service)))),
            app_info_fn=None,
            verbosity=0,
        )

        scheduler.set_estimator()
        
        scheduler.run()
        tock = datetime.now()

        objectives = self.__get_objective_value(scheduler, True)

        solution.objectives = objectives[:]

        return solution

    def __get_objective_value(self, scheduler, estimate):
        jct = list()
        pred_error = list()
        unfairness = list()

        app_list = scheduler._app_list

        for app_id in app_list:
            app = app_list[app_id]

            actual_jct = (app.end_time - app.submit_time).total_seconds()
            jct.append(actual_jct)

            if estimate and len(app.estimated_end_time) > 0:
                estimated_jct = (
                    app.estimated_end_time[0] - app.submit_time
                ).total_seconds()
                pred_error.append(
                    100.0 * abs(estimated_jct - actual_jct) / estimated_jct
                )

            num_apps_seen_diff = app.num_apps_seen[0] / app.num_apps_seen[1]
            divided_cluster_size = scheduler._max_capacity / num_apps_seen_diff
            fair_jct = app.service / min(divided_cluster_size, app.initial_demand)
            unfairness.append(max(0, ((actual_jct / fair_jct) - 1.0)))

        # 0, 1, 2
        jct.sort()
        pred_error.sort()
        unfairness.sort()

        # jct, unfairness, pred error

        obj_vals = list()

        for objective in self.objectives:
            if "pred" in objective.get_name():
                # print(f"{objective.get_name()}: {objective(pred_error)}")
                obj_vals.append(objective(pred_error))

            elif "jct" in objective.get_name():
                obj_vals.append(objective(jct))
                # print(f"{objective.get_name()}: {objective(jct)}")

            elif "unfairness" in objective.get_name():
                obj_vals.append(objective(unfairness))
                # print(f"{objective.get_name()}: {objective(unfairness)}")

        return obj_vals

    def get_name(self) -> str:
        return "Boost"


def comp_thresholds(job_sizes, cov_thresh=1.0):
    thresholds = []

    n = 0
    mu = 0
    s = 0
    s2 = 0
    cov = 0

    class_num = 0
    cov_history = list()

    for i in range(len(job_sizes)):
        xn = job_sizes[i]
        s += xn
        s2 += xn * xn
        n += 1
        mu = ((n - 1.0) * mu + xn) / n
        var = (1.0 / n) * (s2 + (n * mu * mu) - (mu * 2.0 * s))

        cov = var / (mu * mu)

        cov_history.append(cov)

        if cov > cov_thresh:
            thresholds.append(int(job_sizes[i - 1]))
            n = 0
            mu = 0
            s = 0
            s2 = 0
            class_num += 1

    thresholds = thresholds + [float("inf")]
    return thresholds, cov, cov_history


def parse_objectives(objectives):
    objectives_list = list()

    allowed_metrics = ["pred_error", "jct", "unfairness"]

    if len(objectives) < 1:
        print("At least one objective must be specified")
        raise NotImplementedError

    for objective in objectives:
        measure, *metric = objective.split("_")

        metric = "_".join(metric)

        if metric not in allowed_metrics:
            print(f"Currently only {allowed_metrics} metrics allowed")
            raise NotImplementedError

        if measure == "avg":
            objectives_list.append(Objective(np.mean, objective))
        elif measure[0] == "p":
            print(float(measure[1:]))
            objectives_list.append(
                Objective(partial(np.percentile, q=float(measure[1:])), objective)
            )
        else:
            print(f"Currently only avg or pXX measure allowed")
            raise NotImplementedError

    return objectives_list




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runs SPEA2 to find Pareto-optimal Boost configs"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-trace", help="trace name", choices=TRACES, type=str)
    group.add_argument("-workload", help="workload name", choices=WORKLOADS, type=str)
    parser.add_argument(
        "-models",
        choices=["realistic", "toy", "linear"],
        default="linear",
        help="which model type to use",
    )

    parser.add_argument("-max_eval", type=int, help="max_eval", default=1280)
    parser.add_argument("-total_gpus", type=int, help="total_gpus", default=64)
    parser.add_argument("-load", type=float, help="load", default=0.8)
    parser.add_argument("-num_apps", type=int, help="num_apps", default=2000)
    parser.add_argument(
        "-fname", type=str, help="name of output pkl file", default=None
    )
    parser.add_argument("-checkpoint", type=int, help="checkpoint (1/0)", default=0)

    parser.add_argument(
        "-objectives", nargs="+", help="list of objectives", type=str, required=True
    )
    parser.add_argument(
        "-population_size", help="size of population", type=int, default=70
    )

    args = parser.parse_args()

    
    load = args.load
    total_gpus = args.total_gpus
    num_apps = args.num_apps
    max_eval = args.max_eval
    fname = args.fname
    checkpoint = args.checkpoint

    objectives = parse_objectives(args.objectives)

    models = Models(args.models)

    seed = 4567

    app_list = {}
    event_queue = list()

    if args.trace:
        gen_workload_from_trace(
            args.trace, app_list, event_queue, models, max_apps=args.num_apps
        )
    else:
        gen_workload(
            args.workload,
            args.workload,
            args.workload,
            args.workload,
            args.load,
            args.total_gpus,
            args.num_apps,
            seed,
            app_list,
            event_queue,
            models,
        )

    problem = BoostTune(total_gpus, app_list, event_queue, objectives)


    fname = fname or f"learnt_configs_{problem.get_name()}.pkl"

    experiment_info = {
        "objectives": objectives,
        "time": datetime.now(),
        "load": load,
        "total_gpus": total_gpus,
        "workload": args.trace if args.trace else args.workload,
    }

    if fname in os.listdir():
        print(f"warning - overwriting {fname}")
        ans = input("Do you want to proceed: Y/N?")

        if ans == "N":
            sys.exit(1)

    with open(fname, "wb") as fp:
        pickle.dump(experiment_info, fp)

    algorithm = SPEA2(
        problem=problem,
        population_size=args.population_size,
        offspring_population_size=int(args.population_size * 1.5),
        mutation=PolynomialMutation(
            probability=1.0 / problem.number_of_variables, distribution_index=20
        ),
        checkpoints=[2**i for i in range(8, 1 + int(np.log(max_eval) / np.log(2)))]
        if checkpoint
        else [],
        fname=fname,
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_eval),
        population_evaluator=RayEvaluator(processes=40),
    )

    tick = datetime.now()
    print(f"Started at {tick}")
    algorithm.run()
    solutions = get_non_dominated_solutions(algorithm.get_result())

    print("done")
    print(f"took {(datetime.now() - tick).total_seconds()} sec")
