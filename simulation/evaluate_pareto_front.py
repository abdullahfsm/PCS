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
from utils.file_name_funcs import extract_common
import matplotlib.pyplot as plt

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
            prio_func=lambda a: (a.submit_time-tick).total_seconds() - ((1.0/gamma) * math.log(1.0/(1.0-math.exp(-1.0*gamma*a.estimated_service)))),
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

    common_file_terms = extract_common(files)

    with open(f"evaluated_pareto_front_{common_file_terms}.pkl",'wb') as fp:
        pickle.dump(scheduler_stats, fp)




def plot_avg_jct_avg_pred_error(file):
    with open(file,'rb') as fp:
        scheduler_stats = pickle.load(fp)

    boost = scheduler_stats['learnt_configs_BOOST_avg_jct_avg_pred_error_gavel.pkl'] 
    wfq =  scheduler_stats['learnt_configs_WFQTuneWoHeuristics_avg_jct_avg_pred_error_gavel.pkl']
    

    # avg jct vs avg pred_error

    wfq_avg_jct = [np.mean(w['jct']) for w in wfq]
    wfq_avg_pred_error = [np.mean(w['pred_error']) for w in wfq]


    boost_avg_jct = [np.mean(w['jct']) for w in boost]
    boost_avg_pred_error = [np.mean(w['pred_error']) for w in boost]


    min_jct = min(boost_avg_jct+wfq_avg_jct)

    wfq_avg_jct = [w/min_jct for w in wfq_avg_jct]
    boost_avg_jct = [w/min_jct for w in boost_avg_jct]


    plt.scatter(wfq_avg_jct,wfq_avg_pred_error,label='WFQ')
    # plt.scatter(boost_avg_jct,boost_avg_pred_error,label='BOOST')

    plt.xlabel('avg jct')
    plt.ylabel('avg pred_error')

    plt.show()


# evaluated_pareto_front_avg_jct_avg_pred_error_themis1.pkl
def plot_pareto_curve(file):
    plot_avg_jct_avg_pred_error(file)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-files",
        help="Pareto front/result file names",
        nargs="+",
    )
    args = parser.parse_args()

    if len(args.files) == 1 and "evaluated" in args.files[0]:
        plot_pareto_curve(args.files[0])
    else:
        main(args.files)




