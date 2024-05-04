import sys

import copy
import numpy as np
from heapq import heapify, heappop, heappush
from datetime import datetime, timedelta
import math


from .common import Event, App, Job
from .GenericScheduler import AppGenericScheduler


class AppThemisScheduler(AppGenericScheduler):
    """docstring for AppThemisScheduler"""
    def __init__(self, total_gpus, event_queue, app_list, quantum=1, app_info_fn="results.csv", suppress_print=False, verbosity=1, p_error=None):
        super(AppThemisScheduler, self).__init__(total_gpus, event_queue, app_list, app_info_fn, suppress_print, verbosity, p_error)

        self._redivision_event_queue = list()
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

    def __pick_min_heap(self, heap1, heap2):

        if len(heap1) == 0:
            return heap2
        elif len(heap2) == 0:
            return heap1
        
        heap1_event = heap1[0]
        heap2_event = heap2[0]

        if heap1_event < heap2_event:
            return heap1
        else:
            return heap2


    def redivision(self, event):
        self._redivision_event_queue = list()

        next_redivision = self._quantum
        
        heappush(self._redivision_event_queue, Event(event_id=0, event_time=event.event_time + timedelta(seconds=float(next_redivision)), event_type="REDIVISION"))                

    def run(self, cond=lambda: False):

        while len(self._event_queue) > 0 or len(self._end_event_queue) > 0:

            event = heappop(self.__pick_min_heap(self.__pick_min_heap(self._event_queue, self._end_event_queue), self._redivision_event_queue))

            self.progress_active_apps(event.event_time)            
            self._last_event_time = event.event_time

            self.report_progress(event)

            if event.event_type == Event.APP_SUB:
                self.handle_app_sub_event(event)

            elif event.event_type == Event.JOB_END:
                self.handle_job_end_event(event)

            if event.event_type in [Event.APP_SUB, Event.JOB_END, "REDIVISION"]:
                self.redivision(event)

            self.update_allocations(event.event_time)

            self.update_end_events(event.event_time)

            if event.event_type == Event.APP_SUB and self._estimate and np.random.uniform() < 1.2:
                self.sim_estimate(app = self._app_list[event.app_id])

            if cond():
                break


