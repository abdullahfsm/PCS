import dill as pickle
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sim_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'../simulation'))
data_path = os.path.abspath(os.path.dirname(__file__))

sys.path.append(sim_path)


def load_scheduler(policy, trace, parsed=False):
    fname = os.path.join(data_path, f'{policy}_{trace}_result_scheduler.pkl')
    
    if parsed:
        fname = fname.replace('.pkl', '_parsed.pkl')

    with open(fname, 'rb') as fp:
        scheduler = pickle.load(fp)
    return scheduler

def save_scheduler(scheduler, policy, trace):
    fname = os.path.join(data_path, f'{policy}_{trace}_result_scheduler_parsed.pkl')
    with open(fname, 'wb') as fp:
        pickle.dump(scheduler, fp)

def main():
    schedulers = {}
    policies = ['SRSF', 'AFS', 'THEMIS', 'PCS_jct', 'PCS_bal', 'PCS_pred']
    trace = 'ee9e8c'
    for policy in policies:
        scheduler = load_scheduler(policy, trace)
        
        num_attempts_t = {
            0: [],
            1: [],
            10: [],
            40: [],
            60: [],
            100: [],
        }

        num_attempts_t = {
            10: [],
            20: [],
            100: [],
            200: [],
            
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
            fracX = []

            app.errors = np.array([100.0*abs(app.jct - predjct)/predjct for predjct in app.predjcts])
            if app.errors[0] < 10.0:
                continue
            
            for tick in app.jctpred_ticks:
                fracX.append(100.0*(tick - app.submit_time).total_seconds()/app.jct)    
            
            
            assert(len(fracX) == len(app.errors))
            
            for t in num_attempts_t:
                index = np.where(app.errors <= t)[0][0]
                num_attempts_t[t].append(fracX[index])
        scheduler.num_attempts_t = num_attempts_t
        save_scheduler(scheduler, policy, trace)
        del scheduler

if __name__ == '__main__':
    main()