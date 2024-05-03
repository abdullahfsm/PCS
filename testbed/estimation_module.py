import sys
from heapq import heappush, heappop, heapify






def app_to_pt(gpus, delta):
    

    assert(len(gpus) == len(delta))

    total_service = sum(map(lambda g, t: g * t, gpus, delta))

    t_service = [total_service]

    for i in range(len(gpus)):
        t_service.append(t_service[-1] - (gpus[i] * delta[i]))

    p = lambda l1,l2, t: 0.5 * t * (l1+l2)
    
    t_priority=0

    for i in range(len(t_service)-1):
        t_priority += p(t_service[i], t_service[i+1], delta[i])
    return t_priority

def within(src, target):
    if abs(src - target) < 1e-5:
        return True
    return False

class App(object):
    """docstring for App"""
    def __init__(self, app_id=0, submit_time=0, total_stages=1, prio=None, deadline=float('INF'),status="future"):
        super(App, self).__init__()
        self._app_id = app_id
        self._submit_time = submit_time
        self._total_stages = total_stages
        self._stage = 0
        self._jobs = [[] for _ in range(int(total_stages))]
        self._start_time = float('INF')
        self._end_time = float('INF')
        self._demand = None
        self._prio = prio
        self._status = status
        self._duration = None
        self._remaining_time = None
        self._remaining_service = None
        self._deadline = deadline

        if prio is None:
            self._prio = submit_time

    @property
    def deadline(self):
        return self._deadline

    @deadline.setter
    def deadline(self, deadline):
        self._deadline = deadline
        
    @property
    def remaining_time(self):
        return self._remaining_time

    @remaining_time.setter
    def remaining_time(self, remaining_time):
        self._remaining_time = remaining_time

    @property
    def remaining_service(self):
        return self._remaining_service

    @remaining_service.setter
    def remaining_service(self, remaining_service):
        self._remaining_service = remaining_service
    

    @property
    def prio(self):
        return self._prio
    
    @prio.setter
    def prio(self, prio):
        self._prio = prio

    @property
    def stage(self):
        return self._stage

    @property
    def total_stages(self):
        return self._total_stages
    
    @property
    def status(self):
        return self._status
    
    @status.setter
    def status(self, status):
        self._status = status
        
    @property
    def demand(self):
        # demand based on current stage max demand
        app_demand = 0
        for job in self._jobs[self._stage]:
            app_demand += job.demand
        self._demand = app_demand
        return self._demand
    
    @property
    def app_id(self):
        return self._app_id
    
    @property
    def submit_time(self):
        return self._submit_time
    
    @property
    def jobs(self):
        return self._jobs

    @property
    def start_time(self):
        return self._start_time

    @start_time.setter
    def start_time(self, start_time):
        self._start_time = start_time
        

    @property
    def end_time(self):
        return self._end_time

    @end_time.setter
    def end_time(self, end_time):
        self._end_time = end_time
        

    def progress_stage(self):
        self._stage+=1
        
        if self._stage == self._total_stages:
            self._stage = self._total_stages - 1
            return False
        return True
    
    def stage_demand(self, stage):
        stage_demand = 0
        for job in self._jobs[stage]:
            stage_demand += job.demand
        return stage_demand


    def stage_remaining_time(self, stage):
        remaining_times = []
        for job in self._jobs[stage]:
            remaining_times.append(job.remaining_time)
        return max(remaining_times)

    def stage_start_time(self, stage):
        start_time = self.start_time
        for i in range(stage):
            start_time += self.stage_duration(i)
        return start_time

    def stage_duration(self, stage):
        return max(list(map(lambda j: j.duration, self.jobs[stage])))

    def stage_end_time(self, stage):
        return self.stage_start_time(stage) + self.stage_duration(stage)

    def __gt__(self, other):
        return self._prio > other.prio
    def __lt__(self, other):
        return self._prio < other.prio
    def __eq__(self, other):
        return self._prio == other.prio

class Job(object):
    """docstring for Job"""
    def __init__(self, app_id=0, job_id=0, stage_id=0, submit_time=0, demand=1, duration=1, prio=None, status="future"):
        super(Job, self).__init__()
        self._app_id = app_id
        self._job_id = job_id
        self._stage_id = stage_id
        self._submit_time = submit_time
        self._status = status
        self._attempts = list()
        self._demand = demand
        self._duration = duration
        self._remaining_time = duration
        self._remaining_service = demand * duration
        self._last_event_time = submit_time

        self._prio = prio
        if prio is None:
            self._prio = submit_time
    
    @property
    def remaining_time(self):
        return self._remaining_time

    @remaining_time.setter
    def remaining_time(self, remaining_time):
        self._remaining_time = remaining_time

    @property
    def remaining_service(self):
        return self._remaining_service

    @remaining_service.setter
    def remaining_service(self, remaining_service):
        self._remaining_service = remaining_service


    @property
    def last_event_time(self):
        return self._last_event_time

    @last_event_time.setter
    def last_event_time(self, last_event_time):
        self._last_event_time = last_event_time
    
    @property
    def stage_id(self):
        return self._stage_id
    
    @property
    def duration(self):
        return self._duration
    
    @property
    def app_id(self):
        return self._app_id
    
    @property
    def job_id(self):
        return self._job_id
    
    @property
    def submit_time(self):
        return self._submit_time
    
    @property
    def status(self):
        return self._status
    
    @status.setter
    def status(self, status):
        self._status = status

    @property
    def attempts(self):
        return self._attempts
    
    @property
    def demand(self):
        return self._demand
    
    @property
    def prio(self):
        return self._prio

    @prio.setter
    def prio(self, prio):
        self._prio = prio
    
    def __gt__(self, other):
        return self._prio > other.prio
    def __lt__(self, other):
        return self._prio < other.prio
    def __eq__(self, other):
        return self._prio == other.prio

class Event(object):
    """docstring for Event"""
    def __init__(self, event_id, event_time, event_type):
        super(Event, self).__init__()
        self._event_id = event_id
        self._event_time = event_time
        self._event_type = event_type
        self._criteria = event_time


    def __gt__(self, other):
        if self._criteria == other.criteria:
            if ("start" in self._event_type or "sub" in self._event_type) and "end" in other._event_type:
                return True
            elif "end" in self._event_type and ("start" in other._event_type or "sub" in other._event_type):
                return False
            else:
                return self._event_id > other._event_id
        return (self._criteria > other._criteria)
    def __lt__(self, other):
        if self._criteria == other.criteria:
            if ("start" in self._event_type or "sub" in self._event_type) and "end" in other._event_type:
                return False
            elif "end" in self._event_type and ("start" in other._event_type or "sub" in other._event_type):
                return True
            else:
                return self._event_id < other._event_id
        return (self._criteria < other._criteria)
    def __eq__(self, other):
        return (self._criteria == other._criteria)

    @property
    def event_id(self):
        return self._event_id
    
    @property
    def event_time(self):
        return self._event_time
    
    @property
    def event_type(self):
        return self._event_type
    
    @property
    def criteria(self):
        return self._criteria
    
'''
class Logger(object):
    """docstring for Logger"""
    def __init__(self, logging=0):
        super(Logger, self).__init__()
        self._log_cnt = 0
        self._logging=logging
    def log(self, event, status):
                
        if self._logging == 2:
            print("(%d) event_id: %s event_time: %d event_type: %s" % (self._log_cnt, event.event_id, event.event_time, event.event_type))
            print(status)
            print("%d GPUs free" % scheduler.total_gpus)
            print("================================")
            self._log_cnt += 1
        elif self._logging == 1:
            print("\r%d Apps done. %d GPUs free" % (scheduler.num_finished_apps, scheduler.total_gpus),end='')
            if scheduler.total_gpus < 0:
                print("ERROR")
                sys.exit(1)
        else:
            pass
'''

class AppPrioScheduler(object):
    """This class implements a Prio Scheduler for Apps"""
    # Non work conserving
    def __init__(self, total_gpus, scheduling_policy):
        super(AppPrioScheduler, self).__init__()
        self._max_capacity = total_gpus
        self._avail_capacity = total_gpus
        self._scheduling_policy = scheduling_policy

        self._wait_queue = list()
        self._active_apps = list()
        self._num_finished_jobs = 0
        self._num_finished_apps = 0
        self._current_event_queue = None
        self._future_event_queue = None
        self._app_list = None
        self._job_list = None
        self._cur_time = None


        print("Init App %s Scheduler with %d GPUs" % (scheduling_policy, total_gpus))


    def set_params(self, current_event_queue, future_event_queue, app_list, job_list, event_time):
        self._current_event_queue = current_event_queue
        self._future_event_queue = future_event_queue
        self._app_list = app_list
        self._job_list = job_list
        self._cur_time = event_time

    def __get_event_queue(self, event_time):
        if event_time <= self._cur_time:
            return self._current_event_queue
        return self._future_event_queue

    def __update_remaining_time(self, event_time, app):


        remaining_time = 0

        for job in app._jobs[app.stage]:

            if len(job.attempts) > 0:
            
                job.remaining_time -= (event_time - job.last_event_time)
                job.last_event_time = event_time
            else:
                print("Maybe i shouldn't be here")
                job.remaining_time = job.duration



        for stage in range(app.stage, app.total_stages):
            stage_remaining_time = []
            for job in app._jobs[stage]:
                stage_remaining_time.append(job.remaining_time)
            remaining_time += max(stage_remaining_time)
        app.remaining_time = remaining_time

    def __update_remaining_service(self, event_time, app, ep='MAX'):

        remaining_time = 0

        for job in app._jobs[app.stage]:
            job.remaining_service = job.demand * job.remaining_time

        if self.scheduling_policy == "SRSF-MAX":
            app.remaining_service = app.demand * app.remaining_time
        else:
            remaining_service = 0
            for stage in range(app.stage, app.total_stages):
                for job in app._jobs[stage]:
                    remaining_service += job.remaining_service
            app.remaining_service = remaining_service

    def __progress_active_apps(self, event_time):
        for app in self._active_apps:

            self.__update_remaining_time(event_time, app)
            self.__update_remaining_service(event_time, app)



            if self._scheduling_policy == "FIFO":
                app.prio = -1 * app.submit_time
            elif self._scheduling_policy == "SRTF":
                app.prio = -1 * app.remaining_time
            elif self._scheduling_policy == "SRSF-MAX":
                app.remaining_service = app.remaining_time * app.demand
                app.prio = -1 * app.remaining_service
            elif self._scheduling_policy == "SRSF-AWARE":
                app.prio = -1 * app.remaining_service
            elif self._scheduling_policy == "STF":
                app.prio = -1 * app.duration
            elif self._scheduling_policy == "SF":
                app.prio = -1 * app.demand
            elif self._scheduling_policy == "SRSF-PT":
                gpus = []
                delta = []
                for stage in range(app.stage, app.total_stages):
                    gpus.append(app.stage_demand(stage))
                    delta.append(app.stage_remaining_time(stage))
                app.prio = -1 * app_to_pt(gpus, delta)
            elif self._scheduling_policy == "EDF":
                app.prio = -1 * app.deadline
            else:
                print("INVALID SP")
                sys.exit(1)        

        heapify(self._active_apps)

    def __preempt_app(self, event):
        preempted_app = heappop(self._active_apps)

        preempted_app.prio = -1 * preempted_app.prio
        preempted_app.status = "preempted"

        for preempted_job in preempted_app.jobs[preempted_app.stage]:
            
            event_queue = self.__get_event_queue(preempted_job.attempts[-1]["end_time"])
            
            preempted_job.attempts[-1]["end_time"] = event.event_time
            preempted_job.status = "preempted"

            for i, e in enumerate(event_queue):

                if preempted_job.job_id == e.event_id and e.event_type == "job_end":

                    event_queue.pop(i)
                    heapify(event_queue)
                    break

        heappush(self._wait_queue, preempted_app)
        self._avail_capacity += preempted_app.demand
        return preempted_app.app_id


    # remove apps whose deadline has elapsed
    def __remove_apps(self, event):

        num_apps_removed = 0

        for i, t in enumerate(self._active_apps):

            if event.event_time > t.deadline:
                removed_app = self._active_apps.pop(i)
                removed_app.status = "removed"

                num_apps_removed +=1

                for removed_job in removed_app.jobs[removed_app.stage]:

                    event_queue = self.__get_event_queue(removed_job.attempts[-1]["end_time"])

                    removed_job.attempts[-1]["end_time"] = float('inf')
                    removed_job.status = "removed"

                    for j, e in enumerate(event_queue):

                        if removed_job.job_id == e.event_id and e.event_type == "job_end":
                            event_queue.pop(j)
                            heapify(event_queue)
                            break

                self._avail_capacity += removed_app.demand
        
        if num_apps_removed > 0:
            heapify(self._active_apps)
            self.__empty_wait_queue(event.event_time)

        return num_apps_removed
        
    def handle_app_sub_event(self, event):



        self.__progress_active_apps(event.event_time)

        app = self._app_list[event.event_id]
        app.status = "sub"
        for stage in range(app.total_stages):
            for job in app.jobs[stage]:
                job.status = "sub"

        status = None


        if app.demand <= self._avail_capacity and len(self._wait_queue) == 0:
            self.__start_appstage(app, event.event_time)
            status = "app successfully submitted"
        else:

            
            if self._scheduling_policy == "EDF":
                self.__remove_apps(event)
            

            # Unclear what the right heuristic should be

            candidate_apps = list(filter(lambda t: app.prio < (-1 * t.prio), self._active_apps))
            potential_availability = sum(list(map(lambda t: t.demand, candidate_apps)))
            preempted_appids = list()

            if potential_availability >= app.demand:
                while self._avail_capacity < app.demand:
                    preempted_appid = self.__preempt_app(event)
                    preempted_appids.append(preempted_appid)

                preempted_appids = ",".join(list(map(str, preempted_appids)))

                    
                '''
                if app.prio < (-1 * self._active_apps[0].prio) and app.demand <= self._active_apps[0].demand:
                preempted_app = heappop(self._active_apps)

                preempted_app.prio = -1 * preempted_app.prio
                preempted_app.status = "preempted"

                for preempted_job in preempted_app.jobs[preempted_app.stage]:
                    preempted_job.attempts[-1]["end_time"] = event.event_time
                    preempted_job.status = "preempted"

                    for i, e in enumerate(event_queue):

                        if preempted_job.job_id == e.event_id and e.event_type == "job_end":

                            event_queue.pop(i)
                            heapify(event_queue)
                            break

                heappush(self._wait_queue, preempted_app)
                self._avail_capacity += preempted_app.demand
                '''




                assert(self._avail_capacity >= app.demand)
                self.__start_appstage(app, event.event_time)
                # status = "app_id %d preempted. app_id %d started" % (preempted_app.app_id, app.app_id)

                status = "app_ids %s preempted. app_id %d started" % (preempted_appids, app.app_id)
            else:
                app.status = "queued"
                heappush(self._wait_queue, app)
                status = "app pushed in wait queue"

        return status

    # have to look at this
    def __empty_wait_queue(self, event_time):
        while len(self._wait_queue) > 0:
            
            waiting_app = self._wait_queue[0]
            
            if waiting_app.demand <= self._avail_capacity:
                waiting_app = heappop(self._wait_queue)
                waiting_app.prio = -1 * waiting_app.prio
                self.__start_appstage(waiting_app, event_time)
            else:
                return "end->wait_queue"

    # have to look at this
    def __start_appstage(self, app, event_time):
        self._avail_capacity -= app.demand
    
        if app.status == "sub" or app.status == "queued":
            app.start_time = event_time

        if app.status != 'active':
            app.status = 'active'
            app.end_time = event_time + app.remaining_time
            # only add to active app if first stage
            heappush(self._active_apps, app)
            
        for job in app.jobs[app.stage]:
            job.attempts.append({"start_time": event_time, "end_time": event_time + job.remaining_time})
            job.prio = event_time
            job.status = "active"
            job.last_event_time = event_time
            

            event_queue = self.__get_event_queue(event_time + job.remaining_time)
            heappush(event_queue, Event(event_id=job.job_id, event_time=event_time + job.remaining_time, event_type="job_end"))

    def handle_job_end_event(self, event):

        self.__progress_active_apps(event.event_time)

        job = self._job_list[event.event_id]


        # if not (within(job.remaining_time, 0.0)):
        #     print("job_id: %d. time: %f. remaining_time: %f" % (job.job_id, event.event_time, job.remaining_time))

        assert(within(job.remaining_time, 0.0)), print(job.remaining_time)
        

        job.status = "end"
        self._num_finished_jobs += 1

        app = self._app_list[job.app_id]

        job_statuses = map(lambda j: j.status == "end", app.jobs[app.stage])

        # end of a stage
        if all(job_statuses):

            # because stage has ended
            self._avail_capacity += app.demand

            if app.progress_stage():
                self.__start_appstage(app, event.event_time)
            else:
                app.status = 'end'

                # remove from active_apps
                for i, t in enumerate(self._active_apps):
                    if t.app_id == app.app_id:
                        self._active_apps.pop(i)
                        heapify(self._active_apps)
                        break


                    
                self._num_finished_apps += 1

            self.__empty_wait_queue(event.event_time)
        else:
            pass

            

        return "job_end->stage_inprog"


    def handle_unknown_event(self, event):
        return "unknown"

    def num_waiting_apps(self):
        return len(self._wait_queue)

    def num_active_apps(self):
        return len(self._active_apps)

    @property
    def scheduling_policy(self):
        return self._scheduling_policy
    

    @property
    def total_gpus(self):
        return self._avail_capacity

    @property
    def max_gpus(self):
        return self._max_capacity
    

    @property
    def num_finished_jobs(self):
        return self._num_finished_jobs    

    @property
    def num_finished_apps(self):
        return self._num_finished_apps


class Estimator(object):
    """docstring for Estimator"""
    def __init__(self, scheduling_policy, total_gpus):
        super(Estimator, self).__init__()
        

        self._scheduling_policy = scheduling_policy
        self._total_gpus = total_gpus
        self._future_event_queue = None
        self._current_event_queue = None
        self._app_list = None
        self._job_list = None
        self._cur_time = None

        self._scheduler = AppPrioScheduler(total_gpus, scheduling_policy)

    def __create_events(self):

        for app_id in self._app_list:

            app = self._app_list[app_id]

            duration = 0        
            service = 0
            for stage in range(app.total_stages):
                stage_duration = []
                for job in app.jobs[stage]:
                    stage_duration.append(job.duration)
                    service += (job.duration * job.demand)
                duration += max(stage_duration)
            app.remaining_time = duration
            app.duration = duration
            app.remaining_service = service

            if self._scheduling_policy == "FIFO":
                app.prio = app.submit_time
            elif self._scheduling_policy == "SRTF":
                app.prio = app.remaining_time
            elif self._scheduling_policy == "SRSF-MAX":
                app.remaining_service = app.remaining_time * app.demand
                app.prio = app.remaining_service
            elif self._scheduling_policy == "SRSF-AWARE":
                app.prio = app.remaining_service
            elif self._scheduling_policy == "STF":
                app.prio = app.duration
            elif self._scheduling_policy == "SF":
                app.prio = app.demand
            elif self._scheduling_policy == "SRSF-PT":
                gpus = []
                delta = []
                for stage in range(app.stage, app.total_stages):
                    gpus.append(app.stage_demand(stage))
                    delta.append(app.stage_remaining_time(stage))
                app.prio = app_to_pt(gpus, delta)
            elif self._scheduling_policy == "EDF":
                app.prio = app.deadline
            else:
                print("INVALID SP")
                sys.exit(1)        

            if app.status == "sub":
                heappush(self._future_event_queue, Event(event_id=app_id, event_time=app.submit_time, event_type="app_sub"))
            else:
                heappush(self._current_event_queue, Event(event_id=app_id, event_time=app.start_time, event_type="app_sub"))


        

    # list_of_apps -> self._app_list, self._job_list
    def __read_list_of_apps(self, list_of_apps):


        # app trace should follow app_id,total_stages,submit_time,job_id,num_gpu,_,stage_id,_,duration,deadline format
       
        for entry in list_of_apps:


            app_id,total_stages,submit_time,job_id,num_gpu,status,stage_id,event_time,duration,_ = entry


            job = Job(app_id=int(app_id), job_id=int(job_id), stage_id=int(stage_id), demand=int(num_gpu), duration=max(float(duration), 1.0), prio=None, status=status)
            job.remaining = duration

            if int(app_id) not in self._app_list:
                self._app_list[int(app_id)] = App(app_id=int(app_id), submit_time=float(submit_time), total_stages=int(total_stages), status=status)
                if status != "sub":
                    self._app_list[int(app_id)].start_time = float(event_time)


            self._app_list[int(app_id)].jobs[int(stage_id)].append(job)
            self._job_list[int(job_id)] = job


    def get_estimate(self, list_of_apps, this_app_id, event_time):
        

        self._job_list = {}
        self._app_list = {}
        self._future_event_queue = list()
        self._current_event_queue = list()



        self.__read_list_of_apps(list_of_apps)
        self.__create_events()
        self._scheduler.set_params(self._current_event_queue, self._future_event_queue, self._app_list, self._job_list, event_time)
        


    

        while len(self._current_event_queue) + len(self._future_event_queue) > 0:

            if len(self._current_event_queue) > 0:
                event = heappop(self._current_event_queue)
            else:
                event = heappop(self._future_event_queue)


            if event.event_type == "app_sub":

                status = self._scheduler.handle_app_sub_event(event)

            elif event.event_type == "job_end":
                
                status = self._scheduler.handle_job_end_event(event)

            else:
                status = self._scheduler.handle_unknown_event(event)
                sys.exit(1)

        if self._scheduler.num_waiting_apps() > 0:
            print("Cluster is not big enough for largest job")
            sys.exit(1)

        this_app = self._app_list[this_app_id]
        return [this_app_id, this_app.start_time, this_app.end_time]


        


    
if __name__ == '__main__':
    pass