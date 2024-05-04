import sys

import copy
import numpy as np

import math

import warnings

from GenericScheduler import AppGenericScheduler
from common import Event, App, Job


EPS=1e-7

class AppAFSScheduler(AppGenericScheduler):
    """docstring for AppAFSScheduler"""
    def __init__(self, total_gpus, event_queue, app_list, app_info_fn="results.csv", suppress_print=False, verbosity=1, p_error=None):
        super(AppAFSScheduler, self).__init__(total_gpus, event_queue, app_list, app_info_fn, suppress_print, verbosity, p_error)
        print("Warning. compute_remaining_time makes 1job assumption")
                


    def alg_c_concept(self, total_gpus):
        js = [m for m in self._active_apps]
        for m in js:
            m.tmp_gpus = 0
        gpus = total_gpus

        while gpus > 0 and len(js) > 0:
            cand = js[0]
            for m in js[1:]:
                if self.compute_remaining_time(m, m.tmp_gpus) < self.compute_remaining_time(cand, cand.tmp_gpus):
                    l, h = m, cand
                else:
                    l, h = cand, m


                sl0 = 1 / (self.compute_remaining_time(l, l.tmp_gpus) + EPS)
                sl1 = 1 / (self.compute_remaining_time(l, l.tmp_gpus+1)+ EPS)
                sh0 = 1 / (self.compute_remaining_time(h, h.tmp_gpus)+ EPS)
                sh1 = 1 / (self.compute_remaining_time(h, h.tmp_gpus+1)+ EPS)


                if sl0 == 0:
                    cand = l if sl1 > sh1 else h
                elif (sl1 - sl0) / sl0 > (sh1 - sh0) / sh1:
                    cand = l
                else:
                    cand = h

                '''
                with warnings.catch_warnings():
                    warnings.filterwarnings('error', category=RuntimeWarning)
                    
                    try:


                    except RuntimeWarning as rw:

                        print(f"Warning: {rw}")

                        print("====================")
                        print("demands ad alloc:")
                        print(l.demand)
                        print(l.tmp_gpus)
                        print(l.remaining_service)
                        print(l.estimated_remaining_service)
                        print(l.status)
                        print(h.demand)
                        print(h.tmp_gpus)
                        print(h.estimated_remaining_service)
                        print(h.remaining_service)
                        print(h.status)
                        print("====================")
                        print("thrpt:")
                        
                        print(l.jobs[0].thrpt(l.tmp_gpus))
                        print(l.jobs[0].thrpt(l.tmp_gpus+1))
                        print(h.jobs[0].thrpt(h.tmp_gpus))
                        print(h.jobs[0].thrpt(h.tmp_gpus+1))
                        print("====================")
                        print("remaining times:")
                        print(self.compute_remaining_time(l, l.tmp_gpus))
                        print(self.compute_remaining_time(l, l.tmp_gpus+1))
                        print(self.compute_remaining_time(h, h.tmp_gpus))
                        print(self.compute_remaining_time(h, h.tmp_gpus+1))

                        print("====================")
                        print("sl's:")
                        print(sl0,sl1,sh0,sh1)
                        print("====================")
                                                
                        sys.exit(1)

                    # except Exception as e:
                    #     print("====================")

                    #     print(l.demand)
                    #     print(h.demand)

                    #     print("====================")
                    #     print(l.tmp_gpus)
                    #     print(l.jobs[0].thrpt(l.tmp_gpus))
                    #     print(h.tmp_gpus)
                    #     print(h.jobs[0].thrpt(h.tmp_gpus))
                    #     print("====================")

                    #     print(self.compute_remaining_time(l, l.tmp_gpus))
                    #     print(self.compute_remaining_time(l, l.tmp_gpus+1))
                    #     print(self.compute_remaining_time(h, h.tmp_gpus))
                    #     print(self.compute_remaining_time(h, h.tmp_gpus+1))


                    #     print(sl0,sl1,sh0,sh1)
                    #     print("====================")
                    #     raise e
                '''

            cand.tmp_gpus += 1



            if cand.tmp_gpus == cand.demand:
                js.remove(cand)

            gpus -= 1
        

        app_id_to_allocation = {}

        for m in self._active_apps:
            app_id_to_allocation[m.app_id] = m.tmp_gpus
        return app_id_to_allocation


    def compute_remaining_time(self, app, app_current_allocation):

        thrpt = app.jobs[0].thrpt(app_current_allocation)

        if thrpt > 0:
            
            # changed here
            return app.estimated_remaining_service/thrpt
        return float('inf')

        # app_a.remaining_service/current_allocation[app_a.app_id] if current_allocation[app_a.app_id] > 0 else float('inf')

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
                if app_a.estimated_remaining_service < app_b.estimated_remaining_service:
                    a_star = app_a
                else:
                    a_star = app_b
            else:

                # app_a_remaining_time = app_a.remaining_service/current_allocation[app_a.app_id] if current_allocation[app_a.app_id] > 0 else float('inf')
                # app_b_remaining_time = app_b.remaining_service/current_allocation[app_b.app_id] if current_allocation[app_b.app_id] > 0 else float('inf')

                app_a_remaining_time = self.compute_remaining_time(app_a, current_allocation[app_a.app_id])
                app_b_remaining_time = self.compute_remaining_time(app_b, current_allocation[app_b.app_id])

                if app_a_remaining_time >= app_b_remaining_time:
                    app_a, app_b = app_b, app_a

                    # throughput with current allocation
                    p_a, p_b = app_a.jobs[0].thrpt(current_allocation[app_a.app_id]), app_b.jobs[0].thrpt(current_allocation[app_b.app_id])


                    if current_allocation[app_a.app_id] < app_a.demand and current_allocation[app_b.app_id] < app_b.demand:
                        # throughput with extra GPU
                        p_a_p = app_a.jobs[0].thrpt(current_allocation[app_a.app_id]+1)
                        p_b_p = app_b.jobs[0].thrpt(current_allocation[app_b.app_id]+1)

                        if (p_b_p - p_b)/p_b_p > (p_a_p - p_a)/p_a_p:
                            a_star = app_b
                        else:
                            a_star = app_a

                    elif current_allocation[app_a.app_id] == app_a.demand:
                        a_star = app_b
                    elif current_allocation[app_b.app_id] == app_b.demand:
                        a_star = app_a
                    else:
                        print("shouldnt be here")
        return a_star



    def compute_allocation(self, event_time):
    

        total_demand = sum([app.demand for app in self._active_apps])

        residual = min(total_demand, self._max_capacity)


        app_id_to_allocation = self.alg_c_concept(self._max_capacity)

        for app_id in app_id_to_allocation:
            
            allocation = app_id_to_allocation[app_id]

            residual -= allocation
            demand = self._app_list[app_id].demand

            assert(allocation <= demand), (allocation, demand, app_id)


        assert(math.isclose(residual, 0, abs_tol=1e-3)), residual

        # print(app_id_to_allocation)


        
        # total_demand = sum([app.demand for app in self._active_apps])

        # residual = min(total_demand, self._max_capacity)

        # app_id_to_allocation = {}
        
        # for app in self._active_apps:
        #     app_id_to_allocation[app.app_id] = 0

    
        # while residual > 0:

        #     app = self.top_priority(app_id_to_allocation)

        #     allocation = 1 if app_id_to_allocation[app.app_id] < app.demand else 0

        #     app_id_to_allocation[app.app_id] += allocation
            
        #     residual -= allocation


        # assert(math.isclose(residual, 0)), residual


        return app_id_to_allocation

    def compute_allocation_old(self, event_time):
        
        
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


        assert(math.isclose(residual, 0)), residual


        return app_id_to_allocation