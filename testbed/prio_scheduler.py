import copy
from heapq import heapify, heappop, heappush
import random
import sys

import numpy as np
from Event import Event
from heaps import MaxHeap

from helpers import flat_map, swap
from helpers import within


class TaskPrioScheduler(object):
    """This class implements a Prio Scheduler for Tasks"""
    # Non work conserving
    def __init__(self, total_gpus, scheduling_policy, event_queue, task_list, job_list, suppress_print=False):
        super(TaskPrioScheduler, self).__init__()
        self._max_capacity = total_gpus
        self._avail_capacity = total_gpus
        self._wait_queue = MaxHeap(list())
        self._active_tasks = list()
        self._num_finished_jobs = 0
        self._num_finished_tasks = 0
        self._gpu_util = {}
        self._stats_timeline_ticks = list()
        self._stats_timeline_gpu_util = list()
        self._stats_timeline_queue_length = list()
        self._load = []
        self._num_jobs = []
        self._task_queing_delays = []

        self._last_event_time = None
        self._scheduling_policy = scheduling_policy
        self._event_queue = event_queue
        self._end_event_queue = list()
        self._task_list = task_list
        self._job_list = job_list
        self._suppress_print = suppress_print

        # print("Init Task %s Scheduler with %d GPUs" % (scheduling_policy, total_gpus))

    def __preempt_lower_priority(self, event_time):
        remaining_rate = self._max_capacity
        for i, curr_task in enumerate(self._active_tasks):
            task_demand = sum(map(lambda job: job.demand, curr_task.jobs[curr_task.stage]))
            task_min_required = sum(map(lambda j: j._min_gpu, curr_task.jobs[curr_task.stage]))
            # As soon as we find a task that can't run, block on it and don't let anything lower priority run
            if remaining_rate == 0 or task_min_required > min(task_demand, remaining_rate):
                assert(i > 0)
                # Remove all remaining tasks in this class
                for _ in range(len(self._active_tasks) - i):
                    self.__preempt_task(event_time)
                self.__log_active_tasks()
                return
            else:    
                remaining_rate -= min(task_demand, remaining_rate)
    
    def __remaining_rate(self):
        remaining_rate = self._max_capacity
        for curr_task in self._active_tasks:
            task_demand = sum(map(lambda job: job.demand, curr_task.jobs[curr_task.stage]))
            task_min_required = sum(map(lambda j: j._min_gpu, curr_task.jobs[curr_task.stage]))
            assert(task_min_required <= min(task_demand, remaining_rate))
            assert(min(task_demand, remaining_rate) > 0)
            remaining_rate -= min(task_demand, remaining_rate)
        return remaining_rate

    def __rate_for_task(self, task):
        remaining_rate = self._max_capacity
        for curr_task in self._active_tasks:
            task_demand = sum(map(lambda job: job.demand, curr_task.jobs[curr_task.stage]))
            task_min_required = sum(map(lambda j: j._min_gpu, curr_task.jobs[curr_task.stage]))
            assert(task_min_required <= min(task_demand, remaining_rate))
            assert(min(task_demand, remaining_rate) > 0)
            if curr_task.task_id == task.task_id:
                return min(task_demand, remaining_rate)
            else:
                remaining_rate -= min(task_demand, remaining_rate)
        assert(False), "Task not found"


    def __preempt_task(self, event_time):
        # Pop will remove the last item in the list (lowest priority)
        preempted_task = self._active_tasks.pop()
        preempted_task.prio = preempted_task.prio
        preempted_task.status = "preempted"

        for preempted_job in preempted_task.jobs[preempted_task.stage]:
            if preempted_job.status == "active":
              preempted_job.attempts[-1]["end_time"] = event_time
              preempted_job.status = "preempted"

            for i, e in enumerate(self._end_event_queue):

                if preempted_job.job_id == e.event_id and e.event_type == "job_end":

                    self._end_event_queue.pop(i)
                    heapify(self._end_event_queue)
                    break

        self._wait_queue.heappush(preempted_task)
        self._avail_capacity += preempted_task.demand
        return preempted_task.task_id


    def __handle_task_sub_event(self, event):



        app = self._app_list[event.app_id]
        # set submission time
        app.submit_time = event.event_time
        app.status = "sub"
        for job_id in app.job_list:
            job = app.job_list[job_id]
            job.submit_time = event.event_time
            job.status = "sub"




        app_min_required = 1
        remaining_rate = self.__remaining_rate()



        # The wait queue blocks tasks from starting
        if remaining_rate > 0 and app_min_required <= remaining_rate and len(self._wait_queue) == 0:
            self.__start_app(app, event.event_time)
            
        # Block on higher priority tasks that are waiting
        elif len(self._wait_queue) == 0 or app.prio > self._wait_queue[0].prio:
            

            if self._scheduling_policy == "SRTF":
                task_demand = sum(map(lambda job: job.demand, task.jobs[task.stage]))
                min_remaining_time = task.remaining_time / min(task_demand, self._max_capacity)
                task_possible_priority = -1 * min_remaining_time
            else:
                task_possible_priority = task.prio

            candidate_tasks = list(filter(lambda t: t.prio < task_possible_priority, self._active_tasks))
            potential_availability = sum(list(map(lambda t: self.__rate_for_task(t), candidate_tasks)))
            preempted_taskids = list()

            if potential_availability + remaining_rate > 0 and potential_availability + remaining_rate >= task_min_required:

                self.__log_active_tasks()
                self.__start_taskstage(task, event.event_time)
                # status = "task_id %d preempted. task_id %d started" % (preempted_task.task_id, task.task_id)

                status = "task_ids %s preempted. task_id %d started" % (preempted_taskids, task.task_id)
            else:
                task.status = "queued"
                self._wait_queue.heappush(task)
                status = "task pushed in wait queue"
        else:
            task.status = "queued"
            self._wait_queue.heappush(task)
            status = "task pushed in wait queue"

        new_task_list = {}
        new_job_list = {}
        wait_list = list(map(lambda x: x.val, self._wait_queue.h))
        for t in self._active_tasks + wait_list:
          new_task_list[t.task_id] = t
          for job in flat_map(lambda x: x, t._jobs):
            new_job_list[job.job_id] = job

        if not self._suppress_print:
            if random.random() < 0.2:
                to_copy = copy.deepcopy([new_task_list, new_job_list, wait_list, self._active_tasks, self._end_event_queue, task])
                copied = TaskPrioScheduler(self._max_capacity, self._scheduling_policy, list(), to_copy[0], to_copy[1], suppress_print=True)
                copied._wait_queue = MaxHeap(to_copy[2])
                copied._active_tasks = to_copy[3]
                copied._end_event_queue = to_copy[4]
                new_task = to_copy[5]
                copied.__update_end_events(event.event_time)
                copied.run()
                task.estimated_end_time = new_task.end_time
            else:
                task.estimated_end_time = 0

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

    def __update_end_events(self, event_time):
        for task in self._active_tasks:

            rate = self.__rate_for_task(task)
            for job in task._jobs[task.stage]:
                job.rate = rate

                # print(f"Updating end for task {task.task_id} remaining time {job.remaining_time} with rate {rate} current time {event_time}")
                job.attempts[-1]["end_time"] = event_time + (job.remaining_time / rate)
                job.last_event_time = event_time
        
        for i, e in enumerate(self._end_event_queue):
            e.event_time = self._job_list[e.event_id].attempts[-1]["end_time"]

        heapify(self._end_event_queue)

    # have to look at this
    def __empty_wait_queue(self, event_time):
        while len(self._wait_queue) > 0:
            
            waiting_task = self._wait_queue[0]
            
            task_min_required = sum(map(lambda j: j._min_gpu, waiting_task.jobs[waiting_task.stage]))
            remaining_rate = self.__remaining_rate()
            if remaining_rate > 0 and task_min_required <= remaining_rate:
                waiting_task = self._wait_queue.heappop()
                waiting_task.prio = waiting_task.prio
                self.__start_taskstage(waiting_task, event_time)
            else:
                return "end->wait_queue"

    # have to look at this
    def __start_taskstage(self, task, event_time):
        task_demand = sum(map(lambda job: job.demand, task.jobs[task.stage]))
        self._avail_capacity -= min(self._avail_capacity, task_demand)
    
        if task.status == "sub" or task.status == "queued":
            task.start_time = event_time

        did_activate = False
        if task.status != 'active':
            # only add to active task if first stage
            self._active_tasks.append(task)
            self.__update_priorities(event_time)
            did_activate = task in self._active_tasks
        
        if did_activate:
            task.status = 'active'
            rate = self.__rate_for_task(task)
            for job in task.jobs[task.stage]:
                job.attempts.append({"start_time": event_time, "end_time": event_time + (job.remaining_time/rate)})
                job.prio = event_time
                job.status = "active"
                job.last_event_time = event_time
                heappush(self._end_event_queue, Event(event_id=job.job_id, event_time=event_time + (job.remaining_time/rate), event_type="job_end"))

    def __handle_job_end_event(self, event):
        self.__progress_active_tasks(event.event_time)

        job = self._job_list[event.event_id]

        # if not (within(job.remaining_time, 0.0)):
        #     print("job_id: %d. time: %f. remaining_time: %f" % (job.job_id, event.event_time, job.remaining_time))

        task = self._task_list[job.task_id]

        if not within(task.remaining_time, 0):
            print(f"Job remaining time not 0 task: {task.task_id}")
        assert(within(job.remaining_time, 0.0)), job.remaining_time
        

        job.status = "end"
        self._num_finished_jobs += 1

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
                task.end_time = event.event_time
                queing_delay = task.start_time - task.submit_time
                assert(queing_delay >= 0), queing_delay
                self._task_queing_delays.append(queing_delay)

                # remove from active_tasks
                for i, t in enumerate(self._active_tasks):
                    if t.task_id == task.task_id:
                        self._active_tasks.pop(i)
                        break
                    
                self._num_finished_tasks += 1
                if not self._suppress_print:
                    print("\r%d Tasks done" % (self._num_finished_tasks),end='')

            self.__empty_wait_queue(event.event_time)
        else:
            pass
            # DEBUG
            '''
            print("TaskID: %d jobID: %d %ss" % (job_list[event.event_id].task_id, event.event_id, event.event_type.split('_')[1]))
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

    def __handle_unknown_event(self, event):
        return "unknown"

    def num_waiting_tasks(self):
        return len(self._wait_queue)

    def num_active_tasks(self):
        return len(self._active_tasks)


    def run(self):


        while len(self._event_queue) > 0 or len(self._end_event_queue) > 0:
            
            # util_print_heap(self._event_queue)


            if len(self._end_event_queue) != 0 and len(self._event_queue) != 0:

                sub_event = self._event_queue[0]
                end_event = self._end_event_queue[0]

                if end_event < sub_event:
                    event = heappop(self._end_event_queue)
                else:
                    event = heappop(self._event_queue)
            elif len(self._end_event_queue) == 0:
                event = heappop(self._event_queue)
            else:
                event = heappop(self._end_event_queue)                  

            # load = self._max_capacity - self.__remaining_rate()
            # self._load.append((event.event_time, load))
            # self._num_jobs.append((event.event_time, len(self._active_tasks)))

            # print("event_type: %s event_time: %f event_id: %d" % (event.event_type, event.event_time, event.event_id))

            if event.event_type == "task_sub":
                self._load.append(len(self._wait_queue))
                self._num_jobs.append(len(self._active_tasks))
                status = self.__handle_task_sub_event(event)

            elif event.event_type == "job_end":
                
                status = self.__handle_job_end_event(event)

            else:
                status = self.__handle_unknown_event(event)
                sys.exit(1)
            self.__update_end_events(event.event_time)
        # f = open("/tmp/queue-len.txt", "w")
        # for l in self._load:
        #     f.write(f"{l[0]} {l[1]}\n")
        # f.close()
        # f = open("/tmp/system-num-active.txt", "w")
        # for l in self._num_jobs:
        #     f.write(f"{l[0]} {l[1]}\n")
        # f.close()
        if not self._suppress_print:
            print("The average")
            print(sum(self._load) / len(self._load))
            print(sum(self._num_jobs) / len(self._num_jobs))
            print(f"The average queing delay {np.mean(self._task_queing_delays)}")




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
    def num_finished_tasks(self):
        return self._num_finished_tasks
