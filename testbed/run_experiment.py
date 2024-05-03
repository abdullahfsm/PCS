import ray
from ray import tune
import os, sys, csv

sys.path.append(f"{os.path.expanduser('~')}/automl-setup/schedulers")


import numpy as np
import argparse
import matplotlib.pyplot as plt

import pickle

from ray.tune.schedulers.sync_SHA_timed import SyncSHATimedScheduler as SHA
from ray.tune.schedulers.timed_fifo import TimedFIFOScheduler as TimedFIFO
from tensorflow.keras import datasets, layers, models, regularizers, Input

from ray.tune.integration.keras import TuneReportCallback
from filelock import FileLock
from ray.util.queue import Queue
from ray.tune.callback import Callback


from datetime import datetime, timedelta
from time import sleep
import logging


from schedulers.common import App, Job, Event
from schedulers.helpers import gen_data_from_cdf
from RayPrioScheduler import RayAppPrioScheduler
from RayMCScheduler import RayAppMCScheduler
from RayFairScheduler import RayAppFairScheduler
from RayAFSScheduler import RayAppAFSScheduler
from RayThemisScheduler import RayAppThemisScheduler

from fractions import Fraction as frac


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


def train_cifar10(config, checkpoint_dir=None):

    import tensorflow as tf

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
    logging.getLogger('tensorflow').setLevel(logging.FATAL)


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


@ray.remote
def tune_cifar10(app, event_queue, inactivity_time):


    '''
    sched=SHA(time_attr='time_total_s',
            metric='mean_accuracy',
            budget=app.budget,
            mode="max",
            num_samples=app.num_samples,
            reduction_factor=app.reduction_factor,
            temporal_increase=False)
    '''

    # sched=TimedFIFO(time_attr='time_total_s',budget=max(10, (app.budget/app.num_samples)))






    # time_attr: str = "training_iteration",
    # budget: float = 100.0,
    # num_samples = 2,
    # reduction_factor=2):

    # trial_scheduler=SHA(time_attr='time_total_s',budget=(app.service),num_samples=app.demand)
    trial_scheduler=TimedFIFO(time_attr='time_total_s',budget=(app.service/app.demand))

    
    # sched = FIFO(stop=ray.tune.stopper.MaximumIterationStopper(budget))


    analysis = tune.run(
        train_cifar10,
        resources_per_trial={"gpu": 1},
        name="app_%d" % app.app_id,
        trial_name_creator=lambda T: "app_%d_%s" % (app.app_id, T.trial_id),
        scheduler=trial_scheduler,
        num_samples=app.demand,
        config={"p1": tune.choice([0]),
                "p2": tune.choice([0]),
                "p3": tune.choice([0]),
                "p4": tune.choice([0]),
                "p5": tune.choice([0])},
        event_queue=event_queue,
        event_creator=Event,
        # callbacks=[JobEvent(app, queue)],
        scheduler_trial_runner_queue=app.trial_runner_queue,
        inactivity_time=inactivity_time)


@ray.remote
def app_generator(app_list, event_queue):

    start_time = datetime.now()
    submit_time = datetime.now()
        
    for app_id in app_list:

        app = app_list[app_id]

        sleep(app.sleep_time)

        event = Event(event_id=app_id, event_time=datetime.now(), event_type=Event.APP_SUB, app_id=app_id)
        event_queue.put(event)


def gen_workload_from_trace(fname, app_list, event_queue):
    # app trace should follow app_id,total_stages,submit_time,job_id,num_gpu,_,stage_id,_,duration,deadline format
    # app list is a dictionary mapping from app_id to object App
    

    with open(fname, 'r') as fp:
        csvReader = csv.reader(fp)
        next(csvReader)
        for row in csvReader:

            app_id,submit_time,service,num_jobs,sleep_time = row

            app_id = int(app_id)
            service = float(service)
            num_jobs = int(num_jobs)
            sleep_time = float(sleep_time)

            jobs = {}

            for job_id in range(num_jobs):

                jobs[job_id] = Job(app_id=app_id, job_id=job_id, service = service/num_jobs,
                                    demand=1,
                                    min_demand=1)

                jobs[job_id].thrpt_dic = [0,1.0]


            '''
            job_id = 0
            jobs[job_id] = Job(app_id=app_id, job_id=job_id, service=service,
                                demand=num_jobs,
                                min_demand=np.random.choice(min_gpus_per_job))
            '''


            app = App(app_id=app_id, jobs=jobs, deadline=None)
            app.exec_func = tune_cifar10
            app.sleep_time = sleep_time
            app_list[app.app_id] = app


            print("\r%d Apps generated" % (app_id+1),end='')
    print("")


    app_generator.remote(app_list, event_queue)


def gen_workload(cdf_app_service_times, cdf_num_jobs_per_app, cdf_max_gpus_per_job, cdf_min_gpus_per_job, load, num_gpus, num_apps, seed, app_list, event_queue):


    np.random.seed(seed)


    file_dir = os.path.dirname(os.path.abspath(__file__))



    app_service_times = gen_data_from_cdf(f"{file_dir}/schedulers/cdfs/cdf-app-service-times-{cdf_app_service_times}.csv",
                                        num_points=num_apps, dtype=int, interpolation=True)
    num_jobs_per_app = gen_data_from_cdf(f"{file_dir}/schedulers/cdfs/cdf-num-jobs-per-app-{cdf_num_jobs_per_app}.csv",
                                        num_points=num_apps, dtype=int, interpolation=True)
    max_gpus_per_job = gen_data_from_cdf(f"{file_dir}/schedulers/cdfs/cdf-max-gpus-per-job-{cdf_max_gpus_per_job}.csv",
                                        num_points=100, dtype=int, interpolation=True)
    min_gpus_per_job = gen_data_from_cdf(f"{file_dir}/schedulers/cdfs/cdf-min-gpus-per-job-{cdf_min_gpus_per_job}.csv",
                                        num_points=100, dtype=int, interpolation=True)
    
    avg_interarrival_time = (np.mean(app_service_times))/((load)*num_gpus)
    sleep_times = [0.0] + list(map(lambda s: int(max(1,s)), np.random.exponential(avg_interarrival_time, num_apps-1)))
    

    submit_time = 0

    with open(f"{file_dir}/workload.csv",'w') as fp:
        fp.write("app_id,submit_time,service,num_jobs,sleep_time\n")
        
        for app_id in range(num_apps):

            num_jobs = num_jobs_per_app[app_id]
            service = max(int(float(app_service_times[app_id])/num_jobs), 30) * num_jobs

            jobs = {}

            for job_id in range(num_jobs):
                jobs[job_id] = Job(app_id=app_id, job_id=job_id, service = (service/num_jobs),
                                    demand=np.random.choice(max_gpus_per_job),
                                    min_demand=np.random.choice(min_gpus_per_job))

        
            app = App(app_id=app_id, jobs=jobs, deadline=None)
            app.exec_func = tune_cifar10            
            app.sleep_time = sleep_times[app_id]
            app_list[app.app_id] = app


            submit_time += app.sleep_time
            fp.write(f"{app.app_id},{submit_time},{app.remaining_service},{len(app.jobs)},{app.sleep_time}\n")


            print("\r%d Apps generated" % (app_id+1),end='')

    print("")

    app_generator.remote(app_list, event_queue)

    
if __name__ == '__main__':



    parser = argparse.ArgumentParser()

    parser.add_argument('-head_ip', help="IP address of head ray node", type=str, default="10.1.1.2")
    parser.add_argument('-head_port', help="port# of head ray node", type=str, default="6379")
    parser.add_argument('-from_trace', help="1/0 to generate workload using trace", type=int, default=0)
    parser.add_argument('-cdf_app_service_times', help = "fname of app service times", type=str, default="small")
    parser.add_argument('-cdf_num_jobs_per_app', help = "fname of num jobs per app", type=str, default="small")
    parser.add_argument('-cdf_max_gpus_per_job', help = "fname of max gpus per job", type=str, default="1GPU")
    parser.add_argument('-cdf_min_gpus_per_job', help = "fname of min gpus per job", type=str, default="0GPU")
    parser.add_argument('-num_apps', help="number of apps to generate", type=int, default=1)

    parser.add_argument('-load', help = "load", type=float, default=0.8)


    parser.add_argument('-scheduling_policy', help="Scheduling policy", type=str, default="MCS")
    parser.add_argument('-logging', help="logging verbosity (0-2)", default=1, type=int)
    parser.add_argument('-estimation_policy', help='estimation_policy', default='MAX', type=str)
    parser.add_argument('-output_file', default="results.csv", type=str)
    parser.add_argument('-seed', type=int, default=4567)

    parser.add_argument('-MCS_config_file', default=None, type=str)

    args = parser.parse_args()


    ray.init(address=f"{args.head_ip}:{args.head_port}", _redis_password="tf_cluster_123")
    total_gpus = ray.cluster_resources()["GPU"]
    scheduling_policy = args.scheduling_policy
    output_file = args.output_file



    app_list = {}
    event_queue = Queue()



    if scheduling_policy == "MCS":


        # class_detail = {"num_classes": 2, "class_thresholds": [500.0, float('inf')], "class_rates": [0.75,0.25]}

        if args.MCS_config_file == None:
            class_detail = {"num_classes": 3, "class_thresholds": [1523, 5088, float('inf')], "class_rates": [frac(889,1000),frac(1,10),frac(11,1000)]}
        else:
            with open(args.MCS_config_file, "rb") as fp:
                class_detail = pickle.load(fp)


        print(class_detail["num_classes"])
        print(class_detail["class_thresholds"])
        print(class_detail["class_rates"])


        scheduler = RayAppMCScheduler(total_gpus=total_gpus,
                                    event_queue=event_queue,
                                    app_list=app_list,
                                    class_detail=class_detail,
                                    quantum=200,
                                    app_info_fn=output_file)
    elif scheduling_policy == "SRTF":


        scheduler = RayAppPrioScheduler(total_gpus=total_gpus,
                                    event_queue=event_queue,
                                    app_list=app_list,
                                    prio_func=lambda a: a.remaining_service/a.demand if a.demand > 0 else a.remaining_service,
                                    app_info_fn=output_file)

    elif scheduling_policy == "SRSF":

        scheduler = RayAppPrioScheduler(total_gpus=total_gpus,
                                    event_queue=event_queue,
                                    app_list=app_list,
                                    prio_func=lambda a: a.remaining_service,
                                    app_info_fn=output_file)

    elif scheduling_policy == "LAS":

        scheduler = RayAppPrioScheduler(total_gpus=total_gpus,
                                    event_queue=event_queue,
                                    app_list=app_list,
                                    prio_func=lambda a: a.service - a.remaining_service,
                                    app_info_fn=output_file)

    elif scheduling_policy == "FIFO":

        scheduler = RayAppPrioScheduler(total_gpus=total_gpus,
                                    event_queue=event_queue,
                                    app_list=app_list,
                                    prio_func=lambda a: a.submit_time,
                                    app_info_fn=output_file)

    elif scheduling_policy == "FS":
        scheduler = RayAppFairScheduler(total_gpus=total_gpus,
                                                event_queue=event_queue,
                                                app_list = app_list,
                                                quantum=200,
                                                app_info_fn=output_file)

    elif scheduling_policy == "AFS":
        scheduler = RayAppAFSScheduler(total_gpus=total_gpus,
                                    event_queue=event_queue,
                                    app_list=app_list,
                                    app_info_fn=output_file)

    elif scheduling_policy == "THEMIS":
        scheduler = RayAppThemisScheduler(total_gpus=total_gpus,
                                    event_queue=event_queue,
                                    app_list=app_list,
                                    quantum=200,
                                    app_info_fn=output_file)
    else:
        raise NotImplementedError



    if args.from_trace:
        gen_workload_from_trace("workload.csv", app_list, event_queue)
    else:
        gen_workload(args.cdf_app_service_times,
                    args.cdf_num_jobs_per_app,
                    args.cdf_max_gpus_per_job,
                    args.cdf_min_gpus_per_job,
                    args.load,
                    total_gpus,
                    args.num_apps,
                    args.seed,
                    app_list,
                    event_queue)


    print("Starting experiment with %d Apps" % len(app_list))

    
    tick = datetime.now()
    scheduler.run()
    tock = datetime.now()
    

    sleep(10.0)


    if scheduler._num_finished_apps != len(app_list):
        print("Cluster is not big enough for largest job")
        sys.exit(1)


    print("\nExpt ended.")
