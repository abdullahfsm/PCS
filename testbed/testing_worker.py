import ray
from ray import tune
import numpy as np
import argparse
import matplotlib.pyplot as plt
from heapq import heappush, heappop, heapify

# from ray.tune.schedulers.hyperband import HyperBandScheduler as HB
from ray.tune.schedulers.sync_successive_halving import SyncSuccessiveHalving as SHA
from ray.tune.schedulers.timed_fifo import TimedFIFOScheduler as TimedFIFO
from ray.tune.schedulers.trial_scheduler import FIFOScheduler as FIFO
from tensorflow.keras import datasets, layers, models, regularizers, Input
from ray.tune.integration.keras import TuneReportCallback
from filelock import FileLock
import os
from ray.util.queue import Queue
from ray.tune.callback import Callback

from app_generator import gen_SHA_app


from datetime import datetime
from datetime import timedelta
from time import sleep
import sys



from common import Event, App, Job
from estimation_module import Estimator


from mcs_scheduler import AppMCScheduler
from rate_prio_scheduler import AppPrioScheduler



def event_creator(event_id, event_type, event_time, job_id, app_id):
    return Event(event_id=event_id, event_type=event_type, event_time=event_time, job_id=job_id, app_id=app_id)


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



class AppCallbacks(Callback):
            """docstring for AppCallbacks"""
            def __init__(self, arg):
                super(AppCallbacks, self).__init__()
                self.arg = arg
                        


@ray.remote
def tune_cifar10(app, queue=None, scheduler_trial_runner_queue=None):


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
    sched=TimedFIFO(time_attr='time_total_s',budget=(app.budget/app.num_samples))

    
    # sched = FIFO(stop=ray.tune.stopper.MaximumIterationStopper(budget))


    analysis = tune.run(
        train_cifar10,
        resources_per_trial={"gpu": 1},
        name="app_%d" % app.app_id,
        trial_name_creator=lambda T: "app_%d_%s" % (app.app_id, T.trial_id),
        scheduler=sched,
        num_samples=app.num_samples,
        config={"p1": tune.choice([0]),
                "p2": tune.choice([0]),
                "p3": tune.choice([0]),
                "p4": tune.choice([0]),
                "p5": tune.choice([0])},
        event_queue=queue,
        event_creator=event_creator,
        # callbacks=[JobEvent(app, queue)],
        scheduler_trial_runner_queue=scheduler_trial_runner_queue)


    return analysis

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=2)
    parser.add_argument("--reduction_factor", type=int, default=2)
    parser.add_argument("--budget", type=float, default=20)
    args = parser.parse_args()


    ray.init(address="10.1.1.2:6379", _redis_password="tf_cluster_123")


    remaining_ids = list()

    scheduler_trial_runner_queue = Queue()
    event_queue = Queue()


    app = App(app_id=0, num_samples=args.num_samples, budget=args.budget, reduction_factor=1, 
            total_stages=1, sleep_time=None, prio=0)


    remaining_ids.append(tune_cifar10.remote(app,
                        queue=event_queue,
                        scheduler_trial_runner_queue=scheduler_trial_runner_queue))




    scheduler_trial_runner_queue.put(0)
    sleep(20)
    scheduler_trial_runner_queue.put(10)
    # sleep(20)
    # scheduler_trial_runner_queue.put(10)
    



    analysis, *_ = ray.get(remaining_ids)




    print("Best config: ", analysis.get_best_config(
        metric="mean_accuracy", mode="max"))

