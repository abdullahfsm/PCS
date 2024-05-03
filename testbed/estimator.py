import os, sys, argparse
from heapq import heappush, heappop, heapify
import numpy as np
import copy, bisect
import matplotlib.pyplot as plt
import pickle


def task_to_pt(gpus, delta):
    

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
    def __init__(self, task_id=0, submit_time=0, total_stages=1, prio=None, deadline=float('INF'),status="future"):
        super(App, self).__init__()
        self._task_id = task_id
        self._submit_time = submit_time
        self._total_stages = total_stages
        self._stage = 0
        self._jobs = [[] for _ in range(int(total_stages))]
        self._start_time = float('INF')
        self._end_time = float('INF')
        self._demand = None
        self._prio = prio
        self._status = status
        self._estimated_start_time = []
        self._estimated_end_time = []
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
    def estimated_start_time(self):
        return self._estimated_start_time[-1]
    
    @estimated_start_time.setter
    def estimated_start_time(self, estimated_start_time):
        self._estimated_start_time.append(estimated_start_time) 

    @property
    def estimated_end_time(self):
        return self._estimated_end_time[-1]

    @estimated_end_time.setter
    def estimated_end_time(self, estimated_end_time):
        self._estimated_end_time.append(estimated_end_time)

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
        task_demand = 0
        for job in self._jobs[self._stage]:
            task_demand += job.demand
        self._demand = task_demand
        return self._demand
    
    @property
    def task_id(self):
        return self._task_id
    
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
        self.estimated_start_time = start_time

    @property
    def end_time(self):
        return self._end_time

    @end_time.setter
    def end_time(self, end_time):
        self._end_time = end_time
        self.estimated_end_time = end_time

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
    def __init__(self, task_id=0, job_id=0, stage_id=0, submit_time=0, demand=1, duration=1, prio=None, status="future"):
        super(Job, self).__init__()
        self._task_id = task_id
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
    def task_id(self):
        return self._task_id
    
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
    
def read_task_trace_data(fname="toy_tasks.csv", delim=','):
    # task trace should follow task_id,total_stages,submit_time,job_id,num_gpu,_,stage_id,_,duration,deadline format
    # task list is a dictionary mapping from task_id to object App
    # job list is a dictionary mapping from job_id to object Job

    # self._task_id = task_id
    # self._submit_time = submit_time
    # self._stage = 0
    # self._jobs = jobs
    # self._end_time = None


    job_list = {}
    task_list = {}

    with open(fname, 'r') as fp:
        

        data = fp.readlines()

        task_id,total_stages,submit_time,job_id,num_gpu,status,stage_id,_,duration,imp_task_id = data[0].rstrip().split(delim)
        task_trace_data = data[1:]
   
    for i, task_data in enumerate(task_trace_data):

        if "#" in task_data:
            continue

        # app_id, total_stages,submit_time,job_id,num_gpu,_,stage_id,_,duration,sleep_time
        task_id,total_stages,submit_time,job_id,num_gpu,status,stage_id,event_time,duration,_ = task_data.rstrip().split(delim)


        
        job = Job(task_id=int(task_id), job_id=int(job_id), stage_id=int(stage_id), demand=int(num_gpu), duration=max(float(duration), 1.0), prio=None, status=status)
        job.remaining = duration

        if int(task_id) not in task_list:
            task_list[int(task_id)] = App(task_id=int(task_id), submit_time=float(submit_time), total_stages=int(total_stages), status=status)

            if status != "sub":
                task_list[int(task_id)].start_time = float(event_time)





        task_list[int(task_id)].jobs[int(stage_id)].append(job)
        job_list[int(job_id)] = job
    return task_list, job_list, int(imp_task_id)


def create_state_events():
    state_event_queue = list()


    for task_id in task_list:

        task = task_list[task_id]

        if task.status == "sub":
            continue

        duration = 0        
        service = 0
        for stage in range(task.total_stages):
            stage_duration = []
            for job in task.jobs[stage]:
                stage_duration.append(job.duration)
                service += (job.duration * job.demand)
            duration += max(stage_duration)
        task.remaining_time = duration
        task.duration = duration
        task.remaining_service = service

        if scheduling_policy == "FIFO":
            task.prio = task.submit_time
        elif scheduling_policy == "SRTF":
            task.prio = task.remaining_time
        elif scheduling_policy == "SRSF-MAX":
            task.remaining_service = task.remaining_time * task.demand
            task.prio = task.remaining_service
        elif scheduling_policy == "SRSF-AWARE":
            task.prio = task.remaining_service
        elif scheduling_policy == "STF":
            task.prio = task.duration
        elif scheduling_policy == "SF":
            task.prio = task.demand
        elif scheduling_policy == "SRSF-PT":
            gpus = []
            delta = []
            for stage in range(task.stage, task.total_stages):
                gpus.append(task.stage_demand(stage))
                delta.append(task.stage_remaining_time(stage))
            task.prio = task_to_pt(gpus, delta)
        elif scheduling_policy == "EDF":
            task.prio = task.deadline
        else:
            print("INVALID SP")
            sys.exit(1)        
        
        heappush(state_event_queue, Event(event_id=task_id, event_time=task_list[task_id].start_time, event_type="task_sub"))
    return state_event_queue[:]


def create_submit_events():
    event_queue = list()


    for task_id in task_list:
        task = task_list[task_id]


        if task.status != "sub":
            continue


        duration = 0        
        service = 0
        for stage in range(task.total_stages):
            stage_duration = []
            for job in task.jobs[stage]:
                stage_duration.append(job.duration)
                service += (job.duration * job.demand)
            duration += max(stage_duration)
        task.remaining_time = duration
        task.duration = duration
        task.remaining_service = service

        if scheduling_policy == "FIFO":
            task.prio = task.submit_time
        elif scheduling_policy == "SRTF":
            task.prio = task.remaining_time
        elif scheduling_policy == "SRSF-MAX":
            task.remaining_service = task.remaining_time * task.demand
            task.prio = task.remaining_service
        elif scheduling_policy == "SRSF-AWARE":
            task.prio = task.remaining_service
        elif scheduling_policy == "STF":
            task.prio = task.duration
        elif scheduling_policy == "SF":
            task.prio = task.demand
        elif scheduling_policy == "SRSF-PT":
            gpus = []
            delta = []
            for stage in range(task.stage, task.total_stages):
                gpus.append(task.stage_demand(stage))
                delta.append(task.stage_remaining_time(stage))
            task.prio = task_to_pt(gpus, delta)
        elif scheduling_policy == "EDF":
            task.prio = task.deadline
        else:
            print("INVALID SP")
            sys.exit(1)        
        
        heappush(event_queue, Event(event_id=task_id, event_time=task_list[task_id].submit_time, event_type="task_sub"))
    return event_queue[:]

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
            print("\r%d Apps done. %d GPUs free" % (scheduler.num_finished_tasks, scheduler.total_gpus),end='')
            if scheduler.total_gpus < 0:
                print("ERROR")
                sys.exit(1)
        else:
            pass


def gen_output_file(fname="my_sim_task.csv", scheduler=None):
    

    with open(fname, 'w') as fp:
        
        task = task_list[imp_task_id]

        fp.write("app_id,start_time,end_time\n")
        fp.write("%s\n" % (",".join(list(map(str,[imp_task_id, task.start_time, task.end_time])))))


    '''
    with open(fname, 'w') as fp:
        fp.write("task_id,num_gpus,submit_time,start_time,end_time,estimated_start_time,estimated_end_time\n")
        for task_id in task_list:
            
            task = task_list[task_id]
            # assert(task.jobs[0][0].attempts[0]["start_time"] == task.estimated_start_time)

            if task.status == 'end' or task.status == 'active':

                fp.write("%s\n" % ",".join(list(map(str,[task.task_id,
                    task.stage_demand(0),
                    task.submit_time,
                    task.start_time,
                    task.end_time,
                    task._estimated_start_time[0],
                    task._estimated_end_time[0],
                    task.deadline,
                    task.status]))))
            elif task.status == 'sub' or task.status == 'removed':
                fp.write("%s\n" % ",".join(list(map(str,[task.task_id,
                    task.stage_demand(0),
                    task.submit_time,
                    None,
                    None,
                    task._estimated_start_time[0],
                    task._estimated_end_time[0],
                    task.deadline,
                    task.status]))))
    '''
    
    '''
    if scheduler is not None:
        total_time = 0
        weighted_time = 0
        for avail_capacity in scheduler._gpu_util:
            weighted_time += (float(scheduler._max_capacity - avail_capacity)/scheduler._max_capacity) * scheduler._gpu_util[avail_capacity]
            total_time += scheduler._gpu_util[avail_capacity]

        print("gpu utilization: %f" % (weighted_time/total_time))

        if load is not None:
            with open('load-analysis', 'a') as fp:
                fp.write("%0.5f,%0.5f\n" % (float(load), (weighted_time/total_time)))

        with open('gpu-util-data','wb') as fp:
            pickle.dump([scheduler._stats_timeline_gpu_util,scheduler._stats_timeline_ticks], fp)

        with open('queue-length-data','wb') as fp:
            pickle.dump([scheduler._stats_timeline_queue_length,scheduler._stats_timeline_ticks], fp)
    '''
    pass

    
def util_print(tl):
    print("================")
    for t in tl:
        print(t.task_id)
    print("================")

def util_membership(m, s):
    for t in s:
        if m.task_id == t.task_id:
            return True
    return False


class AppPrioScheduler(object):
    """This class implements a Prio Scheduler for Apps"""
    # Non work conserving
    def __init__(self, total_gpus, estimation_policy='MAX', scheduling_policy='FIFO'):
        super(AppPrioScheduler, self).__init__()
        self._max_capacity = total_gpus
        self._avail_capacity = total_gpus
        self._wait_queue = list()
        self._active_tasks = list()
        self._estimated_schedule = list()
        self._num_finished_jobs = 0
        self._num_finished_tasks = 0
        self._gpu_util = {}
        self._stats_timeline_ticks = list()
        self._stats_timeline_gpu_util = list()
        self._stats_timeline_queue_length = list()

        self._last_event_time = None
        self._estimation_policy = estimation_policy
        self._scheduling_policy = scheduling_policy

        print("Init App %s Scheduler with %d GPUs" % (scheduling_policy, total_gpus))


    def __update_remaining_time(self, event_time, task):


        remaining_time = 0

        for job in task._jobs[task.stage]:

            if len(job.attempts) > 0:
                job.remaining_time -= (event_time - job.last_event_time)
                job.last_event_time = event_time
            else:
                print("Maybe i shouldn't be here")
                job.remaining_time = job.duration

        for stage in range(task.stage, task.total_stages):
            stage_remaining_time = []
            for job in task._jobs[stage]:
                stage_remaining_time.append(job.remaining_time)
            remaining_time += max(stage_remaining_time)
        task.remaining_time = remaining_time

    def __update_remaining_service(self, event_time, task, ep='MAX'):

        remaining_time = 0

        for job in task._jobs[task.stage]:
            job.remaining_service = job.demand * job.remaining_time

        if self.scheduling_policy == "SRSF-MAX":
            task.remaining_service = task.demand * task.remaining_time
        else:
            remaining_service = 0
            for stage in range(task.stage, task.total_stages):
                for job in task._jobs[stage]:
                    remaining_service += job.remaining_service
            task.remaining_service = remaining_service

    def __progress_active_tasks(self, event_time):
        for task in self._active_tasks:
            self.__update_remaining_time(event_time, task)
            self.__update_remaining_service(event_time, task, self._estimation_policy)



            if scheduling_policy == "FIFO":
                task.prio = -1 * task.submit_time
            elif scheduling_policy == "SRTF":
                task.prio = -1 * task.remaining_time
            elif scheduling_policy == "SRSF-MAX":
                task.remaining_service = task.remaining_time * task.demand
                task.prio = -1 * task.remaining_service
            elif scheduling_policy == "SRSF-AWARE":
                task.prio = -1 * task.remaining_service
            elif scheduling_policy == "STF":
                task.prio = -1 * task.duration
            elif scheduling_policy == "SF":
                task.prio = -1 * task.demand
            elif scheduling_policy == "SRSF-PT":
                gpus = []
                delta = []
                for stage in range(task.stage, task.total_stages):
                    gpus.append(task.stage_demand(stage))
                    delta.append(task.stage_remaining_time(stage))
                task.prio = -1 * task_to_pt(gpus, delta)
            elif scheduling_policy == "EDF":
                task.prio = -1 * task.deadline
            else:
                print("INVALID SP")
                sys.exit(1)        

        heapify(self._active_tasks)

    def __preempt_task(self, event):
        preempted_task = heappop(self._active_tasks)

        preempted_task.prio = -1 * preempted_task.prio
        preempted_task.status = "preempted"

        for preempted_job in preempted_task.jobs[preempted_task.stage]:
            preempted_job.attempts[-1]["end_time"] = event.event_time
            preempted_job.status = "preempted"

            for i, e in enumerate(event_queue):

                if preempted_job.job_id == e.event_id and e.event_type == "job_end":

                    event_queue.pop(i)
                    heapify(event_queue)
                    break

        heappush(self._wait_queue, preempted_task)
        self._avail_capacity += preempted_task.demand
        return preempted_task.task_id


    # remove tasks whose deadline has elapsed
    def __remove_tasks(self, event):

        num_tasks_removed = 0

        for i, t in enumerate(self._active_tasks):

            if event.event_time > t.deadline:
                removed_task = self._active_tasks.pop(i)
                removed_task.status = "removed"

                num_tasks_removed +=1

                for removed_job in removed_task.jobs[removed_task.stage]:
                    removed_job.attempts[-1]["end_time"] = float('inf')
                    removed_job.status = "removed"

                    for j, e in enumerate(event_queue):

                        if removed_job.job_id == e.event_id and e.event_type == "job_end":
                            event_queue.pop(j)
                            break

                self._avail_capacity += removed_task.demand
        
        if num_tasks_removed > 0:
            heapify(self._active_tasks)
            heapify(event_queue)
            self.__empty_wait_queue(event.event_time)

        return num_tasks_removed
        
    def handle_task_sub_event(self, event):

        self.__progress_active_tasks(event.event_time)

        task = task_list[event.event_id]
        task.status = "sub"
        for stage in range(task.total_stages):
            for job in task.jobs[stage]:
                job.status = "sub"

        status = None


        task.estimated_start_time = event.event_time
        task.estimated_end_time = event.event_time + task.duration


        #DEBUG
        '''
        if len(self._active_tasks) > 0:
            print("Active task_id: %d" % self._active_tasks[0].task_id)
            print("Active task prio: %f" % (-1 * self._active_tasks[0].prio))
            print("This task prio: %f" % task.prio)
        '''

        if task.demand <= self._avail_capacity and len(self._wait_queue) == 0:
            self.__start_taskstage(task, event.event_time)
            status = "task successfully submitted"
        else:

            
            if self._scheduling_policy == "EDF":
                self.__remove_tasks(event)
            

            # Unclear what the right heuristic should be

            candidate_tasks = list(filter(lambda t: task.prio < (-1 * t.prio), self._active_tasks))
            potential_availability = sum(list(map(lambda t: t.demand, candidate_tasks)))
            preempted_taskids = list()

            if potential_availability >= task.demand:
                while self._avail_capacity < task.demand:
                    preempted_taskid = self.__preempt_task(event)
                    preempted_taskids.append(preempted_taskid)

                preempted_taskids = ",".join(list(map(str, preempted_taskids)))

                    
                '''
                if task.prio < (-1 * self._active_tasks[0].prio) and task.demand <= self._active_tasks[0].demand:
                preempted_task = heappop(self._active_tasks)

                preempted_task.prio = -1 * preempted_task.prio
                preempted_task.status = "preempted"

                for preempted_job in preempted_task.jobs[preempted_task.stage]:
                    preempted_job.attempts[-1]["end_time"] = event.event_time
                    preempted_job.status = "preempted"

                    for i, e in enumerate(event_queue):

                        if preempted_job.job_id == e.event_id and e.event_type == "job_end":

                            event_queue.pop(i)
                            heapify(event_queue)
                            break

                heappush(self._wait_queue, preempted_task)
                self._avail_capacity += preempted_task.demand
                '''




                assert(self._avail_capacity >= task.demand)
                self.__start_taskstage(task, event.event_time)
                # status = "task_id %d preempted. task_id %d started" % (preempted_task.task_id, task.task_id)

                status = "task_ids %s preempted. task_id %d started" % (preempted_taskids, task.task_id)
            else:
                task.status = "queued"
                heappush(self._wait_queue, task)
                status = "task pushed in wait queue"

        self.__log_cluster_stats(event.event_time)

        # for gpu usage
        if self._last_event_time is None:
            self._last_event_time = event.event_time
        else:
            if self._avail_capacity not in self._gpu_util:
                self._gpu_util[self._avail_capacity] = 0
            self._gpu_util[self._avail_capacity] += (event.event_time - self._last_event_time)
            self._last_event_time = event.event_time

        return status

    # have to look at this
    def __empty_wait_queue(self, event_time):
        while len(self._wait_queue) > 0:
            
            waiting_task = self._wait_queue[0]
            
            if waiting_task.demand <= self._avail_capacity:
                waiting_task = heappop(self._wait_queue)
                waiting_task.prio = -1 * waiting_task.prio
                self.__start_taskstage(waiting_task, event_time)
            else:
                return "end->wait_queue"

    # have to look at this
    def __start_taskstage(self, task, event_time):
        self._avail_capacity -= task.demand
    
        if task.status == "sub" or task.status == "queued":
            task.start_time = event_time

        if task.status != 'active':
            task.status = 'active'
            task.end_time = event_time + task.remaining_time
            # only add to active task if first stage
            heappush(self._active_tasks, task)
            
        for job in task.jobs[task.stage]:
            job.attempts.append({"start_time": event_time, "end_time": event_time + job.remaining_time})
            job.prio = event_time
            job.status = "active"
            job.last_event_time = event_time
            heappush(event_queue, Event(event_id=job.job_id, event_time=event_time + job.remaining_time, event_type="job_end"))

    def handle_job_end_event(self, event):

        self.__progress_active_tasks(event.event_time)

        job = job_list[event.event_id]

        # if not (within(job.remaining_time, 0.0)):
        #     print("job_id: %d. time: %f. remaining_time: %f" % (job.job_id, event.event_time, job.remaining_time))

        assert(within(job.remaining_time, 0.0)), print(job.remaining_time)
        

        job.status = "end"
        self._num_finished_jobs += 1

        task = task_list[job.task_id]

        job_statuses = map(lambda j: j.status == "end", task.jobs[task.stage])

        # end of a stage
        if all(job_statuses):

            # stats for gpu util
            self.__log_cluster_stats(event.event_time)

            # because stage has ended
            self._avail_capacity += task.demand

            if task.progress_stage():
                self.__start_taskstage(task, event.event_time)
            else:
                task.status = 'end'

                # remove from active_tasks
                for i, t in enumerate(self._active_tasks):
                    if t.task_id == task.task_id:
                        self._active_tasks.pop(i)
                        heapify(self._active_tasks)
                        break

                # remove from estimated schedule
                if self._estimation_policy != 'NONE':
                    # REMOVED
                    for i, t in enumerate(self._estimated_schedule):
                        if t.task_id == task.task_id:
                            self._estimated_schedule.pop(i)
                            break
                    
                self._num_finished_tasks += 1

            self.__empty_wait_queue(event.event_time)
        else:
            pass
            # DEBUG
            '''
            print("AppID: %d jobID: %d %ss" % (job_list[event.event_id].task_id, event.event_id, event.event_type.split('_')[1]))
            print(event.event_time)
            print("============================")
            '''

            

        return "job_end->stage_inprog"


    def __log_cluster_stats(self, event_time):
        if self._last_event_time is None:
            self._last_event_time = event_time
            self._stats_timeline_ticks.append(event_time)
        else:
            if self._avail_capacity not in self._gpu_util:
                self._gpu_util[self._avail_capacity] = 0    
            self._gpu_util[self._avail_capacity] += (event_time - self._last_event_time)
            self._stats_timeline_ticks.append(event_time)
            
            self._stats_timeline_gpu_util.append(1.0 - (float(self._avail_capacity)/self._max_capacity))
            self._stats_timeline_queue_length.append(len(self._wait_queue))
            self._last_event_time = event_time

    def handle_unknown_event(self, event):
        return "unknown"

    def num_waiting_tasks(self):
        return len(self._wait_queue)

    def num_active_tasks(self):
        return len(self._active_tasks)

    @property
    def scheduling_policy(self):
        return self._scheduling_policy
    
    @property
    def estimation_policy(self):
        return self._estimation_policy
    

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
    def num_finished_tasks(self):
        return self._num_finished_tasks

    
if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('-task_trace_file', help = "task trace file", type=str, default="task_workload.csv")
    parser.add_argument('-scheduling_policy', help="Scheduling policy", type=str, default="FIFO")
    parser.add_argument('-logging', help="logging verbosity (0-2)", default=1, type=int)
    parser.add_argument('-num_gpus', help='num_gpus', default=512, type=int)
    parser.add_argument('-estimation_policy', help='estimation_policy', default='MAX', type=str)
    parser.add_argument('-output_file', default="sim_result.csv", type=str)
    parser.add_argument('-alpha', default=None, type=int)
    parser.add_argument('-load', default=None, type=float)
    args = parser.parse_args()

    scheduling_policy = args.scheduling_policy
    logger = Logger(0)
    # logger = Logger(args.logging)
    task_trace_file = args.task_trace_file
    total_gpus=args.num_gpus
    estimation_policy = args.estimation_policy
    task_list, job_list, imp_task_id = read_task_trace_data(task_trace_file)
    
    state_event_queue = create_state_events()
    event_queue = create_submit_events()
    load = args.load
    alpha = args.alpha

    output_file = args.output_file

    scheduler = AppPrioScheduler(total_gpus, estimation_policy=estimation_policy, scheduling_policy=scheduling_policy)

    print("Starting sim with %d Apps" % len(event_queue))

    # load scheduler current state
    while len(state_event_queue):
        event = heappop(state_event_queue)
        scheduler.handle_task_sub_event(event)

    print(scheduler._avail_capacity)


    while len(event_queue) > 0:
        event = heappop(event_queue)






        if event.event_type == "task_sub":

            print("%s. task_id: %d. time: %f" % (event.event_type, task_list[event.event_id].task_id, event.event_time))
            status = scheduler.handle_task_sub_event(event)

        elif event.event_type == "job_end":
            
            print("%s. task_id: %d. time: %f" % (event.event_type, job_list[event.event_id].task_id, event.event_time))
            status = scheduler.handle_job_end_event(event)

        else:
            status = scheduler.handle_unknown_event(event)
            sys.exit(1)

        logger.log(event, status)
        
    if scheduler.num_waiting_tasks() > 0:
        print("Cluster is not big enough for largest job")
        sys.exit(1)


    print("\nSim ended. Generating results")
    gen_output_file(output_file, scheduler)
