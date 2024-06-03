import sys
import os
import copy
from datetime import datetime, timedelta
import math
import random
import ray
from GenericScheduler import AppGenericScheduler
from GenericScheduler import multi_runner
from common import Event, App, Job



class AppThemisScheduler(AppGenericScheduler):
    """docstring for AppThemisScheduler"""
    def __init__(self, total_gpus, event_queue, app_list, quantum=1, app_info_fn="results.csv", suppress_print=False, verbosity=1, p_error=None):
        super(AppThemisScheduler, self).__init__(total_gpus, event_queue, app_list, app_info_fn, suppress_print, verbosity, p_error)

        self._redivision_event = None
        self._quantum = quantum

                
    def valuation_function(self, app, event_time, max_allocation):


        if max_allocation > 0:
            observed_contention = app.num_apps_seen[0]/app.num_apps_seen[1]
            divided_cluster_size = self._max_capacity/observed_contention
            t_id = app.service/min(divided_cluster_size, app.initial_demand)
            t_sh = (event_time - app.submit_time).total_seconds() + (app.remaining_service/max_allocation)
            
            return t_sh/t_id
        else:
            return 1.0

    def compute_allocation(self, event_time):
        
        # self._active_apps = sorted(self._active_apps, key=lambda a: self.valuation_function(a, event_time), reverse=True)

        total_demand = sum([app.demand for app in self._active_apps])

        residual = min(total_demand, self._max_capacity)

        app_id_to_allocation = {}

        app_slice = self._active_apps[:]


        for app in self._active_apps:
            app_id_to_allocation[app.app_id] = 0


        while residual > 0:

            app_slice = sorted(app_slice, key=lambda a: self.valuation_function(a, event_time, min(residual, a.demand)), reverse=True)

            top_app = app_slice[0]
            allocation = min(residual, top_app.demand)

            app_id_to_allocation[top_app.app_id] = allocation
            residual -= allocation

            app_slice = app_slice[1:]

        assert(math.isclose(residual, 0)), residual

        return app_id_to_allocation


    def redivision(self, event):
        self._redivision_event = None        
        self._redivision_event = Event(event_id=0, event_time=event.event_time + timedelta(seconds=float(self._quantum)), event_type="REDIVISION")


    def snap_shot(self):
        self._estimator._active_apps = self._active_apps
        self._estimator._last_event_time = self._last_event_time
        self._estimator._app_list = {}

        self._estimator._redivision_event = self._redivision_event
        self._estimator._closest_end_event = self._closest_end_event

        for app in self._estimator._active_apps:
            self._estimator._app_list[app.app_id] = app

        return copy.deepcopy(self._estimator)




    def pick_min_event(self):

        numbers = [self._closest_end_event, self._redivision_event]
        lst = self._event_queue

        inf_event = Event(event_id=-1, event_time=datetime.max, event_type=Event.UNDEFINED)


        numbers = [n if n else inf_event for n in numbers]
        min_answer = min(min(numbers), lst[-1] if lst else inf_event)

        if lst and min_answer.event_type == Event.APP_SUB:
            lst.pop()
        return min_answer


    def scheduler_specific_pre_alloc(self, event):
        if event.event_type in [Event.APP_SUB, Event.JOB_END, "REDIVISION"]:
            self.redivision(event)