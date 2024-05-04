import ray
from RayGenericScheduler import RayAppGenericScheduler
import os, sys
from datetime import datetime, timedelta
from time import sleep
import numpy as np
from functools import partial
import copy


from AFSScheduler import AppAFSScheduler
from common import Event, App, Job


class RayAppAFSScheduler(RayAppGenericScheduler):
    """docstring for RayAppMCScheduler"""
    def __init__(self, total_gpus, event_queue, app_list, app_info_fn="results.csv", suppress_print=True):
        super(RayAppAFSScheduler, self).__init__(total_gpus, event_queue, app_list, app_info_fn, suppress_print=False)
    

    def sim_estimate(self, app):


        # total_gpus, event_queue, app_list, class_detail, app_info_fn="results.csv", suppress_print=False, estimate=False

        snap_shot = AppAFSScheduler(total_gpus=self._max_capacity,
                                    event_queue=[Event(event_id=app.app_id, event_time=datetime.now(), event_type=Event.APP_SUB, app_id=app.app_id)],
                                    app_list={},
                                    app_info_fn=None)


        snap_shot._active_apps = copy.deepcopy(self._active_apps)
        snap_shot._app_id_to_allocation = copy.deepcopy(self._app_id_to_allocation)

        for virtual_app in snap_shot._active_apps+[copy.deepcopy(app)]:
            

            for attr in ['future', 'exec_func', 'trial_runner_queue']:
                if hasattr(virtual_app, attr):
                    delattr(virtual_app, attr)
            
            snap_shot._app_list[virtual_app.app_id] = virtual_app



        snap_shot._estimate = False
        snap_shot._suppress_print = True
        snap_shot._verbosity = 0
        snap_shot._estimator = True

        snap_shot._init_time = self._init_time
        snap_shot._last_event_time = self._last_event_time


        snap_shot.update_end_events(datetime.now())

        
        def break_cond(v_app):
            if v_app.status == App.END:
                return True
            return False
        
        # snap_shot.run()

        snap_shot.run(partial(break_cond, snap_shot._app_list[app.app_id]))
        

        app.update_estimates(snap_shot._app_list[app.app_id].start_time,
                            snap_shot._app_list[app.app_id].end_time)

    def top_priority(self, current_allocation):
        
        while True:
            a_star = np.random.choice(self._active_apps)
            if a_star.demand > 0:
                break


        for app in self._active_apps:
            if app.app_id == a_star.app_id or app.demand == 0:
                continue

            app_a, app_b = a_star, app

            if current_allocation[app_a.app_id] == 0 and current_allocation[app_b.app_id] == 0:
                if app_a.remaining_service < app_b.remaining_service:
                    a_star = app_a
                else:
                    a_star = app_b
            else:

                app_a_remaining_time = app_a.remaining_service/current_allocation[app_a.app_id] if current_allocation[app_a.app_id] > 0 else float('inf')
                app_b_remaining_time = app_b.remaining_service/current_allocation[app_b.app_id] if current_allocation[app_b.app_id] > 0 else float('inf')


                if app_a_remaining_time >= app_b_remaining_time:
                    app_a, app_b = app_b, app_a

                    # throughput with current allocation
                    p_a, p_b = current_allocation[app_a.app_id], current_allocation[app_b.app_id]

                    # throughput with extra GPU
                    p_a_p = p_a+1 if current_allocation[app_a.app_id] < app_a.demand else p_a
                    p_b_p = p_b+1 if current_allocation[app_b.app_id] < app_b.demand else p_b

                    if (p_b_p - p_b)/p_b_p > (p_a_p - p_a)/p_a_p:
                        a_star = app_b
                    else:
                        a_star = app_a
        return a_star



    def compute_allocation(self, event_time):
        
        
        total_demand = sum([app.demand for app in self._active_apps])

        residual = min(total_demand, self._max_capacity)

        app_id_to_allocation = {}
        
        for app in self._active_apps:
            app_id_to_allocation[app.app_id] = 0

    
        while residual > 0:

            app = self.top_priority(app_id_to_allocation)

            allocation = 1 if app_id_to_allocation[app.app_id] < app.demand else 0

            app_id_to_allocation[app.app_id] += allocation
            
            residual -= allocation


        assert(np.isclose(residual, 0)), residual


        return app_id_to_allocation