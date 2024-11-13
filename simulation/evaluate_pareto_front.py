from wfq_tuner import *
from boost_tuner import *

import pandas as pd
import numpy as numpy

import argparse

from MCSScheduler import AppMCScheduler
from PriorityScheduler import AppPrioScheduler

import pickle

import ray
import math
from datetime import datetime
def read_pickle(fname):
    
    with open(fname,'rb') as fp:
        obj = []
        while True:
            try:
                obj.append(pickle.load(fp))
            except Exception as e:
                return obj





def wfq_eval(problem, solutions):
    scheduler_stats = []

    for s in solutions:
        tick = datetime.now()
        class_detail = problem.solution_transformer(s)
        scheduler = AppMCScheduler(
            total_gpus=problem._total_gpus,
            event_queue=copy.deepcopy(problem._event_queue),
            app_list=copy.deepcopy(problem._app_list),
            class_detail=class_detail,
            app_info_fn=None,
            verbosity=0,
        )

        scheduler.set_estimator()        
        scheduler.run()
        scheduler_stats.append(compute_stats(scheduler))
    return scheduler_stats

def boost_eval(problem, solutions):

    scheduler_stats = []

    for s in solutions:
        tick = datetime.now()
        gamma = float(s.variables[0])
        scheduler = AppPrioScheduler(
            total_gpus=problem._total_gpus,
            event_queue=copy.deepcopy(problem._event_queue),
            app_list=copy.deepcopy(problem._app_list),
            prio_func=lambda a: (tick - a.submit_time).total_seconds() - ((1.0/gamma) * math.log(1.0/(1.0-math.exp(-1.0*gamma*a.estimated_service)))),
            app_info_fn=None,
            verbosity=0,
        )

        scheduler.set_estimator()        
        scheduler.run()
        scheduler_stats.append(compute_stats(scheduler))
    return scheduler_stats

def compute_stats(scheduler, estimate=True):
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

    result_dic = {
        "jct": jct,
        "pred_error": pred_error,
        "unfairness": unfairness,
        # "avg_jct": np.mean(jct),
        # "avg_pred_error": np.mean(pred_error),
        # "p99_jct": np.quantile(jct, 0.99),
        # "p99_pred": np.quantile(pred_error, 0.99),
        # "p99.9_jct": np.quantile(jct, 0.999),
        # "p99.9_pred": np.quantile(pred_error, 0.999),
    }

    return result_dic



def main(files):
    
    scheduler_stats = {}

    for f in files:

        obj = read_pickle(f)

        problem = obj[-1]['PROBLEM']
        solutions = obj[-1]['SOLUTIONS']

        if "boost" in f.lower():
            scheduler_stats[f] = boost_eval(problem, solutions)
        else:
            scheduler_stats[f] = wfq_eval(problem, solutions)

    with open("evaluate_pareto_front_results.pkl",'wb') as fp:
        pickle.dump(scheduler_stats, fp)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-files",
        help="Pareto front file names",
        nargs="+",
    )
    args = parser.parse_args()

    main(args.files)




