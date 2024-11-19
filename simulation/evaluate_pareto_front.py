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
import seaborn as sns

# Set the default palette to Seaborn's tab10
# sns.set_palette("tab10")


# Set the global font size
plt.rcParams.update({'font.size': 20})  # Change 14 to your desired size
plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=0.105, right=0.99, top=0.99, bottom=0.12)
palette = sns.color_palette("tab10")

policy_plot_fmt = {
    'WFQ':   {'color': palette[0], 'marker': 'o', 'markersize': 8},
    'BOOST': {'color': palette[1], 'marker': '*', 'markersize': 8},
    'FIFO':  {'color': palette[2], 'marker': '^', 'markersize': 15},
    'SJF':  {'color': palette[3], 'marker': 'v', 'markersize': 15},
}



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
        scheduler_stats.append(compute_stats(scheduler, attribs=class_detail))
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
        scheduler_stats.append(compute_stats(scheduler, attribs=gamma))
    return scheduler_stats

def compute_stats(scheduler, estimate=True, attribs=None):
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
        "attribs": attribs,
        # "avg_jct": np.mean(jct),
        # "avg_pred_error": np.mean(pred_error),
        # "p99_jct": np.quantile(jct, 0.99),
        # "p99_pred": np.quantile(pred_error, 0.99),
        # "p99.9_jct": np.quantile(jct, 0.999),
        # "p99.9_pred": np.quantile(pred_error, 0.999),
    }

    return result_dic


def SRSF_eval(problem):
    scheduler_stats = []


    scheduler = AppPrioScheduler(
        total_gpus=problem._total_gpus,
        event_queue=copy.deepcopy(problem._event_queue),
        app_list=copy.deepcopy(problem._app_list),
        prio_func=lambda a: a.demand * a.estimated_remaining_service/(a.jobs[0].thrpt(a.demand) if len(a.jobs) == 1 else a.demand),
        app_info_fn=None,
        verbosity=0,
    )

    scheduler.set_estimator()        
    scheduler.run()
    scheduler_stats.append(compute_stats(scheduler))
    return scheduler_stats


def SJF_eval(problem):
    scheduler_stats = []


    scheduler = AppPrioScheduler(
        total_gpus=problem._total_gpus,
        event_queue=copy.deepcopy(problem._event_queue),
        app_list=copy.deepcopy(problem._app_list),
        prio_func=lambda a: a.estimated_remaining_service,
        app_info_fn=None,
        verbosity=0,
    )

    scheduler.set_estimator()        
    scheduler.run()
    scheduler_stats.append(compute_stats(scheduler))
    return scheduler_stats



def FIFO_eval(problem):
    scheduler_stats = []

    scheduler = AppPrioScheduler(
        total_gpus=problem._total_gpus,
        event_queue=copy.deepcopy(problem._event_queue),
        app_list=copy.deepcopy(problem._app_list),
        prio_func=lambda a: a.submit_time,
        app_info_fn=None,
        verbosity=0,
    )

    scheduler.set_estimator()        
    scheduler.run()
    scheduler_stats.append(compute_stats(scheduler))
    return scheduler_stats

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

    scheduler_stats['SRSF'] = SRSF_eval(problem)
    scheduler_stats['FIFO'] = FIFO_eval(problem)

    common_file_terms = extract_common(files)

    with open(f"evaluated_pareto_front_{common_file_terms}.pkl",'wb') as fp:
        pickle.dump(scheduler_stats, fp)


def compute_avg_jct_avg_pred_error(schedulers):
    data = []
    for scheduler_name, scheduler in schedulers.items():
        avg_jcts = [np.mean(w['jct']) for w in scheduler]
        avg_pred_errors = [np.mean(w['pred_error']) for w in scheduler]
        # avg_pred_errors = [np.quantile(w['pred_error'],0.90) for w in scheduler]
        
        assert(len(avg_jcts) == len(avg_pred_errors))

        for avg_jct,avg_pred_error in zip(avg_jcts,avg_pred_errors):

            data.append({
                'policy': scheduler_name,
                'avg_jct': avg_jct,
                'avg_pred_error': avg_pred_error,
            })

    data_df = pd.DataFrame(data)

    data_df['norm_avg_jct'] = data_df['avg_jct'] / data_df['avg_jct'].min()

    sorted_df = data_df.sort_values(by='norm_avg_jct')
    

    return sorted_df


def verify_baseline(scheduler_stats):
    update=False    
    obj = None
    problem = None

    for k,v in scheduler_stats.items():
        if not '.pkl' in k:
            continue

        try:
            obj = read_pickle(k)
            problem = obj[-1]['PROBLEM']
            break
        except Exception as e:
            obj=None
            problem = None


    if problem is None:
        raise ValueError("Problem is None")

    if 'SRSF' not in scheduler_stats:        
        scheduler_stats['SRSF'] = SRSF_eval(problem)
        update=True

    if 'SJF' not in scheduler_stats:        
        scheduler_stats['SJF'] = SJF_eval(problem)
        update=True


    if 'FIFO' not in scheduler_stats:
        scheduler_stats['FIFO'] = FIFO_eval(problem)
        update = True

    return update


def plot_avg_jct_avg_pred_error(file):
    with open(file,'rb') as fp:
        scheduler_stats = pickle.load(fp)

    update = verify_baseline(scheduler_stats)
    # update_scheduler_stats
    if update:
        with open(file,'wb') as fp:
            pickle.dump(scheduler_stats, fp)


    boost = None
    wfq = None
    for k in scheduler_stats.keys():
        if 'BOOST' in k:
            boost = scheduler_stats[k]
        elif 'WFQ' in k:
            wfq = scheduler_stats[k]

    assert(not (wfq is None))
    assert(not (boost is None))

    srsf = scheduler_stats['SRSF']
    sjf = scheduler_stats['SJF']
    fifo = scheduler_stats['FIFO']

    df = compute_avg_jct_avg_pred_error({
        'BOOST':boost,
        'SRTF': srsf,
        'SJF': sjf,
        'FIFO': fifo,
        'WFQ': wfq,
    })


    for i, policy in enumerate(['WFQ','BOOST','FIFO','SJF']):

        # plt.scatter(df[df['policy'] == policy]['norm_avg_jct'].tolist(),
        #             df[df['policy'] == policy]['avg_pred_error'].tolist(),label=policy)


        plt.plot(df[df['policy'] == policy]['norm_avg_jct'].tolist(),
                 df[df['policy'] == policy]['avg_pred_error'].tolist(),
                 markevery=1,
                 marker=policy_plot_fmt[policy]['marker'],
                 color=policy_plot_fmt[policy]['color'],
                 markersize=policy_plot_fmt[policy]['markersize'])


    ax = plt.gca()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    alpha=0.25
    if "themis1" in file:
        plt.plot([0.5,1.21],[5,5], color='k', linestyle='--', alpha=alpha, linewidth=2)
        plt.plot([1.125,1.125],[5,-1], color='k', linestyle='--', alpha=alpha, linewidth=2)
        plt.plot([1.21,1.21],[5,-1], color='k', linestyle='--', alpha=alpha, linewidth=2)

    if "gavel" in file:
        plt.plot([0.5,2],[5,5], color='k', linestyle='--', alpha=alpha, linewidth=2)
        plt.plot([1.1,1.1],[5,-1], color='k', linestyle='--', alpha=alpha, linewidth=2)
        plt.plot([2,2],[5,-1], color='k', linestyle='--', alpha=alpha, linewidth=2)


    for policy in ['WFQ','BOOST','FIFO','SJF']:
        plt.plot([500,1000],[5000,5000],
                linewidth=5,
                label=policy,
                marker=policy_plot_fmt[policy]['marker'],
                color=policy_plot_fmt[policy]['color'],
                markersize=15)
        
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
        

    plt.xlabel('Norm. Average JCT')
    plt.ylabel('Average Prediction Error %')


    plt.legend(frameon=False)
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)



    plt.savefig(file.replace('.pkl','.png'),format='png',dpi=300)



    return


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




