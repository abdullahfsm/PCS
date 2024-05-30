import dill as pickle
import os
import sys
import pandas as pd
import numpy as np
from matplotlib.
sim_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'../simulation'))
data_path = os.path.abspath(os.path.dirname(__file__))

sys.path.append(sim_path)


def load_scheduler():
    fname = os.path.join(data_path, 'SRSF_0e4a51_result_scheduler.pkl')
    with open(fname, 'rb') as fp:
        scheduler = pickle.load(fp)
    return scheduler



def main():
    scheduler = load_scheduler()

    num_attempts_t = {
        0: [],
        10: [],
        20: [],
        40: [],
        60: [],
        80: [],
        100: [],
    }

    for app_id, app in scheduler._app_list.items():
        # submit_time = (app.submit_time - scheduler._init_time).total_seconds()
        # start_time = (app.start_time - scheduler._init_time).total_seconds()
        # end_time = (app.end_time - scheduler._init_time).total_seconds()


        num_apps_seen_diff = app.num_apps_seen[0]/app.num_apps_seen[1]

        divided_cluster_size = scheduler._max_capacity/num_apps_seen_diff
        fair_act = app.service/min(divided_cluster_size, app.initial_demand)

        if len(app.estimated_start_time) == 0:            
            continue

        app.jct = (app.end_time - app.submit_time).total_seconds()
        app.predjcts = [(estimated_end_time - app.submit_time).total_seconds() for estimated_end_time in app.estimated_end_time]

        app.errors = np.array([100.0*abs(app.jct - predjct)/predjct for predjct in app.predjcts])
        
        for t in num_attempts_t:
            index = np.where(app.errors <= t)[0][0]
            num_attempts_t[t].append(index)
    scheduler.num_attempts_t = num_attempts_t


if __name__ == '__main__':
    main()