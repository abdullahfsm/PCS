import ray
from ray import tune
import numpy as np
import argparse
import matplotlib.pyplot as plt
from heapq import heappush, heappop, heapify

# from ray.tune.schedulers.hyperband import HyperBandScheduler as HB
from ray.tune.schedulers.sync_successive_halving import SyncSuccessiveHalving as SHA
from ray.tune.schedulers.trial_scheduler import FIFOScheduler as FIFO
from tensorflow.keras import datasets, layers, models, regularizers, Input
from ray.tune.integration.keras import TuneReportCallback
from filelock import FileLock
import os
from ray.util.queue import Queue
from ray.tune.callback import Callback

from datetime import datetime
from datetime import timedelta
from time import sleep
import sys

from app_generator import gen_SHA_app


from common import Event, App, Job
from estimation_module import Estimator

def model_generator(config):
    
    weight_decay = 0.0005


    dropout_scale = 1


    inputs = Input(shape=(32,32,3))

    l = layers.Conv2D(64, (3, 3), padding='same', activation= 'relu',kernel_regularizer=regularizers.l2(weight_decay))(inputs)
    l = layers.BatchNormalization()(l)
    l = layers.Dropout(0.3 * (dropout_scale))(l)

    if config['p1'] == 1:
        l = layers.Conv2D(64, (3, 3), padding='same', activation= 'relu', kernel_regularizer=regularizers.l2(weight_decay))(l)
        l = layers.BatchNormalization()(l)
    
    l = layers.MaxPooling2D(pool_size=(2, 2))(l)


    ###################################################################


    l = layers.Conv2D(128, (3, 3), padding='same', activation= 'relu', kernel_regularizer=regularizers.l2(weight_decay))(l)
    l = layers.BatchNormalization()(l)
    l = layers.Dropout(0.4 * (dropout_scale))(l)

    if config['p2'] == 1:
        l = layers.Conv2D(128, (3, 3), padding='same', activation= 'relu', kernel_regularizer=regularizers.l2(weight_decay))(l)
        l = layers.BatchNormalization()(l)
    
    l = layers.MaxPooling2D(pool_size=(2, 2))(l)


    ###################################################################


    l = layers.Conv2D(256, (3, 3), padding='same', activation= 'relu', kernel_regularizer=regularizers.l2(weight_decay))(l)
    l = layers.BatchNormalization()(l)
    l = layers.Dropout(0.4 * (dropout_scale))(l)

    l = layers.Conv2D(256, (3, 3), padding='same', activation= 'relu', kernel_regularizer=regularizers.l2(weight_decay))(l)
    l = layers.BatchNormalization()(l)
    l = layers.Dropout(0.4 * (dropout_scale))(l)

    if config['p3'] == 1:
        l = layers.Conv2D(256, (3, 3), padding='same', activation= 'relu', kernel_regularizer=regularizers.l2(weight_decay))(l)
        l = layers.BatchNormalization()(l)
    
    l = layers.MaxPooling2D(pool_size=(2, 2))(l)


    ###################################################################


    l = layers.Conv2D(512, (3, 3), padding='same', activation= 'relu', kernel_regularizer=regularizers.l2(weight_decay))(l)
    l = layers.BatchNormalization()(l)
    l = layers.Dropout(0.4 * (dropout_scale))(l)

    l = layers.Conv2D(512, (3, 3), padding='same', activation= 'relu', kernel_regularizer=regularizers.l2(weight_decay))(l)
    l = layers.BatchNormalization()(l)
    l = layers.Dropout(0.4 * (dropout_scale))(l)

    if config['p4'] == 1:
        l = layers.Conv2D(512, (3, 3), padding='same', activation= 'relu', kernel_regularizer=regularizers.l2(weight_decay))(l)
        l = layers.BatchNormalization()(l)
        
    l = layers.MaxPooling2D(pool_size=(2, 2))(l)


    ###################################################################


    l = layers.Conv2D(512, (3, 3), padding='same', activation= 'relu', kernel_regularizer=regularizers.l2(weight_decay))(l)
    l = layers.BatchNormalization()(l)
    l = layers.Dropout(0.4 * (dropout_scale))(l)

    l = layers.Conv2D(512, (3, 3), padding='same', activation= 'relu', kernel_regularizer=regularizers.l2(weight_decay))(l)
    l = layers.BatchNormalization()(l)
    l = layers.Dropout(0.4 * (dropout_scale))(l)

    if config['p5'] == 1:
        l = layers.Conv2D(512, (3, 3), padding='same', activation= 'relu', kernel_regularizer=regularizers.l2(weight_decay))(l)
        l = layers.BatchNormalization()(l)
    
    l = layers.MaxPooling2D(pool_size=(2, 2))(l)


    ###################################################################

    l = layers.Flatten()(l)
    output = layers.Dense(10, activation='softmax')(l)

    model = models.Model(inputs=inputs,outputs=output)

    return model

def normalize(X_train,X_test):
    #this function normalize inputs for zero mean and unit variance
    # it is used when training a model.
    # Input: training set and test set
    # Output: normalized training set and test set according to the trianing set statistics.
    mean = np.mean(X_train,axis=(0,1,2,3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train-mean)/(std+1e-7)
    X_test = (X_test-mean)/(std+1e-7)
    return X_train, X_test

def train_cifar10(config, checkpoint_dir=None):

    import tensorflow as tf


    batch_size = 128
    epochs = 10000
    num_classes = 10

    N_train = 35000
    N_valid = 15000


    with FileLock(os.path.expanduser("~/.data.lock")):
        (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()


    N = (x_train.shape)[0]

    assert(N_train+N_valid == N)

    '''
    x_train = x_train[:5000]
    y_train = y_train[:5000]

    N_train = 4800
    N_valid = 200
    '''

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train, x_test = normalize(x_train, x_test)

    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)


    x_train, y_train = x_train[:N_train], y_train[:N_train]
    x_validation, y_validation = x_train[-N_valid:], y_train[-N_valid:]
    x_test, y_test = x_test, y_test


    ###################################################################

    model = model_generator(config)


    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.CategoricalCrossentropy()


    model.compile(loss=loss,
                optimizer=optimizer,
                metrics=['accuracy'])

    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0,
            validation_data=(x_validation, y_validation),
            callbacks=[TuneReportCallback({
                "mean_accuracy": "accuracy"
            })])

class JobEvent(Callback):
    """docstring for JobEvent"""
    def __init__(self, app, queue):
        super(JobEvent, self).__init__()
        self.queue = queue
        self.app = app
        self.app_started = False
        self.trial_id_to_job_id = {}
        self.job_id_to_trial_id = {}
        self.trial_id_to_status = {}

    def on_trial_complete(self, iteration, trials, trial, **info):

        if self.queue == None:
            return


        print("app_id: %d len(jobs): %d" % (self.app.app_id, len(self.trial_id_to_job_id)))


        if self.trial_id_to_status[trial.trial_id] == "active":
            self.trial_id_to_status[trial.trial_id] = "end"
            job_id = self.trial_id_to_job_id[trial.trial_id]    
            self.queue.put(Event(event_id=self.app.app_id, event_type="job_end", event_time=datetime.now(), job_id=job_id, app_id=self.app.app_id))
            
            


    def on_trial_start(self, iteration, trials, trial, **info):

        if self.queue == None:
            return


        if not self.app_started:
            self.queue.put(Event(event_id=self.app.app_id, event_type="app_start", event_time=datetime.now(), job_id=self.app.app_id, app_id=self.app.app_id))
            self.app_started = True

        for job_id in self.app.job_list:
            if job_id not in self.job_id_to_trial_id:
                self.job_id_to_trial_id[job_id] = trial.trial_id
                self.trial_id_to_job_id[trial.trial_id] = job_id
                self.queue.put(Event(event_id=self.app.app_id, event_type="job_start", event_time=datetime.now(), job_id=job_id, app_id=self.app.app_id))
                self.trial_id_to_status[trial.trial_id] = "active"
                break           

@ray.remote
def tune_cifar10(app, queue=None, scheduler_trial_runner_queue=None):


    sched=SHA(time_attr='time_total_s',
            metric='mean_accuracy',
            budget=app.budget,
            mode="max",
            num_samples=app.num_samples,
            reduction_factor=app.reduction_factor,
            temporal_increase=False)
    
    
    # sched = FIFO(stop=ray.tune.stopper.MaximumIterationStopper(budget))


    analysis = tune.run(
        train_cifar10,
        resources_per_trial={"gpu": 1},
        name="app_%d" % app.app_id,
        scheduler=sched,
        num_samples=app.num_samples,
        config={"p1": tune.choice([0]),
                "p2": tune.choice([0]),
                "p3": tune.choice([0]),
                "p4": tune.choice([0]),
                "p5": tune.choice([0])},
        callbacks=[JobEvent(app, queue)],
        scheduler_trial_runner_queue=scheduler_trial_runner_queue)

    if queue != None:
        queue.put(Event(event_id=app.app_id, event_type="app_end", event_time=datetime.now(), job_id=app.app_id, app_id=app.app_id))


@ray.remote
def app_generator(app_list, queue):

    total_apps = len(app_list)
    generated_apps=0

    while generated_apps < total_apps:


        app = app_list[generated_apps]

        sleep(app._sleep_time)

        queue.put(Event(event_id=app.app_id, event_type="app_sub", event_time=datetime.now(), job_id=app.app_id, app_id=app.app_id))
        
        generated_apps+=1

'''
def start_app(app, queue):
    available_capacity = available_capacity - app.num_samples
    return tune_cifar10.remote(app, queue)
'''
    
def replay_queue(queue):
    while len(queue) > 0:
        event = queue.get(block=True)

        print("=================================")
        print("event_id: %d" % event.event_id)
        print("event_type: %s" % event.event_type)
        print("event_time: %s" % event.event_time)
        print("app_id: %d" % event.app_id)
        print("job_id: %d" % event.job_id)
        print("=================================")


# create estimation module



class AppPrioScheduler(object):
    """This class implements the Prio Scheduler for Apps"""
    # Non work conserving
    def __init__(self, event_queue, warm_up=True, learn_sys_params=True):
        super(AppPrioScheduler, self).__init__()
        self._max_capacity = int(ray.cluster_resources()["GPU"])
        self._avail_capacity = int(ray.available_resources()["GPU"])
        self._wait_queue = list()
        
        self._num_finished_jobs = 0
        self._num_finished_apps = 0
        self._gpu_util = {}
        self._stats_timeline_ticks = list()
        self._stats_timeline_gpu_util = list()
        self._stats_timeline_queue_length = list()
        self._event_queue = event_queue
        self._remaining_ids = list()
        self._init_time = datetime.now()

        self._last_event_time = None

        self._active_apps = list()
        self._dispatched_apps = list()

        self._estimation_overhead = list()
        self._estimator = Estimator(scheduling_policy="FIFO", total_gpus=self._max_capacity)
        self._app_info_fp = {"MAX": None, "MIN": None, "CURRENT": None, "TRUE": None}
        self._job_info_fp = {"MAX": None, "MIN": None, "CURRENT": None, "TRUE": None}

        for estimation_policy in ["MAX", "MIN", "CURRENT", "TRUE"]:
            self._app_info_fp[estimation_policy] = open("app_results_REAL_FIFO_%s.csv" % (estimation_policy),'w')
            self._app_info_fp[estimation_policy].write("app_id,num_samples,submit_time,dispatch_time,start_time,end_time,estimated_start_time,estimated_end_time,theoretical_duration,estimated_duration,actual_duration\n")
            self._job_info_fp[estimation_policy] = open("job_results_REAL_FIFO_%s.csv" % (estimation_policy),'w')
            self._job_info_fp[estimation_policy].write("app_id,job_id,num_gpus,submit_time,dispatch_time,start_time,end_time,theoretical_duration,actual_duration\n")


        if warm_up:
            self.__warm_up()
        self._avg_setup_time_even, self._avg_setup_time_odd, self._avg_duration_delta = self.__learn_sys_params(num_reps = 5, use_prev= not(learn_sys_params))

        print("Init App Prio Scheduler with %d GPUs" % self._max_capacity)

        self._pid_to_gpu_map = {}
        self._app_id_to_trial_runner_queue = {}
        # nvidia-smi --query-compute-apps=pid,gpu_name,gpu_uuid --format=csv

    def __log_app_info(self, app):

        for estimation_policy in ["MAX", "MIN", "CURRENT", "TRUE"]:
            app_info_fp = self._app_info_fp[estimation_policy]
            
            
            app_info_fp.write("%s\n" % ",".join(list(map(str,[app.app_id,
                app.num_samples,
                (app.submit_time - self._init_time).total_seconds(),
                (app.dispatch_time - self._init_time).total_seconds(),
                (app.start_time - self._init_time).total_seconds(),
                (app.end_time - self._init_time).total_seconds(),
                (app.start_estimation[estimation_policy] - self._init_time).total_seconds(),
                (app.end_estimation[estimation_policy] - self._init_time).total_seconds(),
                app.duration.total_seconds(),
                self.__app_execution_time(app),
                (app.end_time - app.start_time).total_seconds()]))))

            job_info_fp = self._job_info_fp[estimation_policy]
            
            for job_id in app.job_list:
                job = app.job_list[job_id]

                job_info_fp.write("%s\n" % ",".join(list(map(str,[app.app_id,
                    job_id,
                    1,
                    (job.submit_time - self._init_time).total_seconds(),
                    (job.dispatch_time - self._init_time).total_seconds(),
                    (job.start_time - self._init_time).total_seconds(),
                    (job.end_time - self._init_time).total_seconds(),
                    app.budget,
                    (job.end_time - job.start_time).total_seconds()]))))
    
    def __estimate_demand(self, app, app_profile, estimation_policy):
        
        if estimation_policy == "TRUE":
            return [app_profile[stage_id][0] for stage_id in range(app.total_stages)]
        elif estimation_policy == "CURRENT":
            return [app.remaining_demand() for stage_id in range(app.total_stages)]
        elif estimation_policy == "MIN":
            return [app_profile[-1][0] for stage_id in range(app.total_stages)]
        elif estimation_policy == "MAX":
            return [app_profile[0][0] for stage_id in range(app.total_stages)]

    def __app_translator(self, this_app, estimation_policy):

        dummy_job_id = 0
        list_of_apps = list()

        for app in self._active_apps:

            app_profile = gen_SHA_app(app.num_samples, app.budget, app.reduction_factor)
            demand_profile = self.__estimate_demand(app, app_profile,estimation_policy)
            total_stages = len(app_profile)

            for stage_id in range(total_stages):

                list_of_apps.append([app.app_id, total_stages, (app.submit_time - self._init_time).total_seconds(),
                                    dummy_job_id, demand_profile[stage_id], app.status, stage_id,
                                    (app.start_time - self._init_time).total_seconds(),
                                    app_profile[stage_id][1] + (self._avg_duration_delta).total_seconds(),-1])
                dummy_job_id+=1


        for app in self._dispatched_apps:

            app_profile = gen_SHA_app(app.num_samples, app.budget, app.reduction_factor)
            demand_profile = self.__estimate_demand(app, app_profile,estimation_policy)
            total_stages = len(app_profile)

            for stage_id in range(total_stages):

                if stage_id == 0:
                    event_time = app_profile[stage_id][1] + (self.__app_setup_time(app) + self._avg_duration_delta).total_seconds()
                else:
                    event_time = app_profile[stage_id][1] + (self._avg_duration_delta).total_seconds()

                list_of_apps.append([app.app_id, total_stages, (app.submit_time - self._init_time).total_seconds(),
                                    dummy_job_id, demand_profile[stage_id], app.status, stage_id,
                                    (app.dispatch_time - self._init_time).total_seconds(),
                                    event_time,-1])
                dummy_job_id+=1

        for app in self._wait_queue + [this_app]:

            app_profile = gen_SHA_app(app.num_samples, app.budget, app.reduction_factor)
            demand_profile = self.__estimate_demand(app, app_profile,estimation_policy)
            total_stages = len(app_profile)


            for stage_id in range(total_stages):

                if stage_id == 0:
                    event_time = app_profile[stage_id][1] + (self.__app_setup_time(app) + self._avg_duration_delta).total_seconds()
                else:
                    event_time = app_profile[stage_id][1] + (self._avg_duration_delta).total_seconds()
                
                list_of_apps.append([app.app_id, total_stages, (app.submit_time - self._init_time).total_seconds(),
                                    dummy_job_id, demand_profile[stage_id], app.status, stage_id,
                                    (app.submit_time - self._init_time).total_seconds(),
                                    event_time,-1])
                dummy_job_id+=1
        return list_of_apps[:]

    def __get_sim_estimate(self, this_app, estimation_policy):

        tick = datetime.now()
        list_of_apps = self.__app_translator(this_app, estimation_policy)
        app_id,start_time,end_time = self._estimator.get_estimate(list_of_apps, this_app.app_id, (this_app.submit_time - self._init_time).total_seconds())
        tock = datetime.now()


        self._estimation_overhead.append((tock - tick).total_seconds())

        assert(app_id == this_app.app_id) 


        estimated_start_time = self.__app_setup_time(this_app) + timedelta(seconds=start_time) + self._init_time
        estimated_end_time = timedelta(seconds=end_time)+self._init_time


        return [estimated_start_time, estimated_end_time]

        
    def __app_setup_time(self, app):
        if (app.app_id // self._max_capacity) % 2 == 0:
            return self._avg_setup_time_even
        return self._avg_setup_time_odd

    def __app_execution_time(self, app):
        return (app.duration + (self._avg_duration_delta * app._total_stages)).total_seconds()


    # added here
    def __preempt_app(self, event):
        preempted_app = heappop(self._active_apps)
        preempted_app.status = "preempted"

    
        self._app_id_to_trial_runner_queue[preempted_app.app_id].put(0)

        # _scheduler_trial_runner_interface.preempt(preempted_app.app_id)

        assert(preempted_app.demand == preempted_app.remaining_demand())

        for job_id in preempted_app.job_list:
            job = preempted_app.job_list[job_id]
            if job.status == "active":
                job.status = "preempted"
        
        self._avail_capacity += preempted_app.demand      
        preempted_app.prio = -1 * preempted_app.prio
        heappush(self._wait_queue, preempted_app)



    def __handle_app_sub_event(self, event):

        app = app_list[event.app_id]


        # set submission time
        app.submit_time = event.event_time
        app.status = "sub"
        for job_id in app.job_list:
            job = app.job_list[job_id]
            job.submit_time = event.event_time
            job.status = "sub"

        
        for estimation_policy in ["MAX", "MIN", "CURRENT", "TRUE"]:
            # estimated_start_time, estimated_end_time = self.__get_sim_estimate(app, estimation_policy)
            estimated_start_time, estimated_end_time = event.event_time, event.event_time
            app.start_estimation[estimation_policy] = estimated_start_time
            app.end_estimation[estimation_policy] = estimated_end_time
        


        if app.demand <= self._avail_capacity and len(self._wait_queue) == 0:
            self.__start_app(app, event.event_time)
        else:

            candidate_apps = list(filter(lambda a: app.prio < (-1 * a.prio), self._active_apps))
            potential_availability = sum(list(map(lambda a: a.demand, candidate_apps)))
            

            if app.demand <= potential_availability:
                while self._avail_capacity < app.demand:
                    self.__preempt_app(event)

                assert(app.demand <= self._avail_capacity)
                self.__start_app(app, event.event_time)
            else:
                heappush(self._wait_queue, app)


    def __handle_app_start_event(self, event):
        app = app_list[event.app_id]
        app.status = "active"            
        app.start_time = event.event_time

        for i, a in enumerate(self._dispatched_apps):
            if a.app_id == app.app_id:
                self._dispatched_apps.pop(i)
                break

        app.prio = -1 * app.prio
        heappush(self._active_apps, app)

    def __handle_app_end_event(self, event):
        app = app_list[event.app_id]
        app.status = "end"
        app.end_time = event.event_time


        for i, a in enumerate(self._active_apps):
            if a.app_id == app.app_id:
                self._active_apps.pop(i)
                heapify(self._active_apps)
                break

        self._num_finished_apps+=1


        self.__log_app_info(app)



    def __handle_job_start_event(self, event):
        app = app_list[event.app_id]
        job = app.job_list[event.job_id]
        job.start_time = event.event_time
        job.status = "active"

    def __handle_job_end_event(self, event):
        app = app_list[event.app_id]
        job = app.job_list[event.job_id]
        job.end_time = event.event_time
        job.status = "end"

        app.demand -= job.demand
        self._avail_capacity += job.demand

        self.__empty_wait_queue(event.event_time)


    def __handle_unknown_event(self, event):
        print("wrong event type")
        sys.exit(1)



    def __start_app(self, app, event_time):
        self._avail_capacity -= app.demand

        # see if this was an app previously preempted
        if app.status == "preempted":
            
            # assign it equal to the demand so it can run
            self._app_id_to_trial_runner_queue[app.app_id].put(app.demand)

            app.status = "active"
            for job_id in app.job_list:
                job = app.job_list[job_id]
                if job.status == "preempted":
                    job.status = "active"

            heappush(self._active_apps, app)
        else:
            app.status = "dispatched"
            self._dispatched_apps.append(app)
            app.dispatch_time = event_time

            for job_id in app.job_list:
                app.job_list[job_id].dispatch_time = event_time
        
            # create interfacing queue
            scheduler_trial_runner_queue = Queue()
            self._app_id_to_trial_runner_queue[app.app_id] = scheduler_trial_runner_queue

            print("started_expt")
            self._remaining_ids.append(tune_cifar10.remote(app, self._event_queue, scheduler_trial_runner_queue=scheduler_trial_runner_queue))
            
            # self._remaining_ids.append(tune_cifar10.remote(app, self._event_queue))



    def __empty_wait_queue(self, event_time):
        
        
        while len(self._wait_queue) > 0:
            
            waiting_app = self._wait_queue[0]
            

            if waiting_app.demand <= self._avail_capacity:
                waiting_app = heappop(self._wait_queue)
                self.__start_app(waiting_app, event_time)
            else:
                break
        

    def __warm_up(self):

        total_gpus = self._max_capacity
        warm_up_ids = list()
        num_warm_up_apps_gen = 2 * total_gpus

        for _ in range(num_warm_up_apps_gen):

            app = App(app_id=0, num_samples=1, budget=20.0, reduction_factor=2, sleep_time=None)

            warm_up_ids.append(tune_cifar10.remote(app))


            if len(warm_up_ids) == total_gpus:
                while warm_up_ids:
                    _, warm_up_ids = ray.wait(warm_up_ids)
                sleep(2.0)
                warm_up_ids = list()
    



    def __learn_sys_params(self, num_reps=5, use_prev=True):

        if use_prev and "parameters.csv" in os.listdir():
            with open("parameters.csv",'r') as fp:
                return list(map(lambda t: timedelta(seconds=float(t)),fp.readlines()[1].rstrip().split(',')))



        temp_queue = Queue()
        temp_app_list = {}
        init_time = datetime.now()
        temp_ids = list()
        total_gpus = self._max_capacity
        num_temp_apps_gen = total_gpus * num_reps



        for app_id in range(num_temp_apps_gen):

            app = App(app_id=app_id, num_samples=1, budget=20.0, reduction_factor=2, sleep_time=None)

            temp_app_list[app_id] = app

            app.submit_time = datetime.now()

            for j in range(app.num_samples):
                app.job_list[j] = Job(app_id=app_id, job_id=j, demand=1)
            temp_ids.append(tune_cifar10.remote(app, temp_queue))

            if len(temp_ids) == total_gpus:
                while temp_ids:
                    _, temp_ids = ray.wait(temp_ids)
                sleep(2.0)
        sleep(3.0)

        # getting state
        while temp_queue:

            event = temp_queue.get()


            app = temp_app_list[event.app_id]

            if event.event_type == "app_start":
                app.start_time = event.event_time
                app.status = "active"
            elif event.event_type == "app_end":
                app.end_time = event.event_time
                app.status = "end"
            elif event.event_type == "job_start" or event.event_type == "job_end":
                pass
            else:
                print("wrong event type")
                sys.exit(1)


        duration_delta = list()
        setup_time_even = list()
        setup_time_odd = list()

        for app_id in temp_app_list:            

            app = temp_app_list[app_id]

            if (app_id // total_gpus) % 2 == 0:
                setup_time_even.append((app.start_time - app.submit_time).total_seconds())
            else:
                setup_time_odd.append((app.start_time - app.submit_time).total_seconds())

            duration_delta.append((app.end_time - app.start_time).total_seconds() - 20.0)

        with open("parameters.csv",'w') as fp:
            fp.write("avg_setup_time_even,avg_setup_time_odd,avg_duration_delta\n")
            fp.write("%f,%f,%f\n" % (np.mean(setup_time_even), np.mean(setup_time_odd), np.mean(duration_delta)))

        return list(map(lambda t: timedelta(seconds=t),[np.mean(setup_time_even), np.mean(setup_time_odd), np.mean(duration_delta)]))



    def run(self, total_apps):
        
        while self._num_finished_apps < total_apps:


            event = self._event_queue.get(block=True)


            if event.event_type == "app_sub":
                self.__handle_app_sub_event(event)
            elif event.event_type == "app_start":
                self.__handle_app_start_event(event)
            elif event.event_type == "app_end":
                self.__handle_app_end_event(event)
            elif event.event_type == "job_start":
                self.__handle_job_start_event(event)
            elif event.event_type == "job_end":
                self.__handle_job_end_event(event)
            else:
                self.__handle_unknown_event(event)



    def num_waiting_apps(self):
        return len(self._wait_queue)    

    @property
    def num_finished_apps(self):
        return self._num_finished_apps



if __name__ == '__main__':



    parser = argparse.ArgumentParser()
    parser.add_argument('-workload', default="workload.csv", help = "workload filename (must be csv)", type=str)

    args = parser.parse_args()


    ray.init(address="10.1.1.2:6379", _redis_password="tf_cluster_123")

    total_capacity = ray.cluster_resources()["GPU"]
    available_capacity = ray.available_resources()["GPU"]



    queue = Queue()

    # create submit events

    app_list = {}

    with open(args.workload) as fp:
        app_data = fp.readlines()[1:]

        for entry in app_data:

            if '#' in entry:
                continue


            app_id,total_stages,submit_time,_,num_samples,_,alpha,_,duration,sleep_time = entry.split(',')


            app_id = int(app_id)
            num_samples = int(num_samples)
            duration = float(duration)
            sleep_time = float(sleep_time)
            alpha=int(alpha)
            total_stages = int(total_stages)

            app = App(app_id=app_id, num_samples=num_samples, budget=duration, reduction_factor=alpha, 
                    total_stages=total_stages, sleep_time=sleep_time, prio=duration*total_stages)
            for j in range(app.num_samples):
                app.job_list[j] = Job(app_id=app_id, job_id=j, demand=1)
            app_list[app_id] = app

    total_apps = len(app_list)

    finished_apps = 0
    remaining_ids = list()
    returned_objs = list()


    sched = AppPrioScheduler(event_queue = queue,
                            learn_sys_params=False,
                            warm_up=False)



    # generate app submission events in the background
    app_generator.remote(app_list, queue)


    tick = datetime.now()

    sched.run(total_apps)
    sleep(5.0)


    remaining_ids = sched._remaining_ids
    while remaining_ids:
        done_ids, remaining_ids = ray.wait(remaining_ids, timeout=200.0)

        if len(done_ids) == 0:
            break


    tock = datetime.now()

    '''
    with open("run_log",'w') as fp:
        fp.write("total_expt_time,total_estimation_time,avg_estimation_time,max_estimation_time\n")
        fp.write("%f,%f,%f,%f\n" % 
            ((tock - tick).total_seconds(), 
            np.sum(sched._estimation_overhead), 
            np.mean(sched._estimation_overhead), 
            max(sched._estimation_overhead)))
    '''
    pass

    # gen_output_file(app_list, fname="app_results.csv", sched_init_time=sched._init_time)