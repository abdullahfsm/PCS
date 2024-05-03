import ray
from ray import tune
import numpy as np
import argparse
import matplotlib.pyplot as plt

# from ray.tune.schedulers.hyperband import HyperBandScheduler as HB
# from ray.tune.schedulers.sync_successive_halving import SyncSuccessiveHalving as SHA
from ray.tune.schedulers.sync_SHA_timed import SyncSHATimedScheduler as SHA
from ray.tune.schedulers.timed_fifo import TimedFIFOScheduler as TimedFIFO
from ray.tune.schedulers.trial_scheduler import FIFOScheduler as FIFO
from tensorflow.keras import datasets, layers, models, regularizers, Input
from ray.tune.integration.keras import TuneReportCallback
from filelock import FileLock
from ray.util.queue import Queue
from ray.tune.callback import Callback


from datetime import datetime, timedelta
from time import sleep
import os, sys, csv
import logging

from schedulers.common import App, Job, Event

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
def tune_cifar10(total_service=400, num_jobs=8, app_id=0):


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
    trial_scheduler=TimedFIFO(time_attr='time_total_s',budget=(total_service/num_jobs))

    
    # sched = FIFO(stop=ray.tune.stopper.MaximumIterationStopper(budget))


    analysis = tune.run(
        train_cifar10,
        resources_per_trial={"gpu": 1},
        name="app_%d" % app_id,
        trial_name_creator=lambda T: "app_%d_%s" % (app_id, T.trial_id),
        scheduler=trial_scheduler,
        num_samples=num_jobs,
        config={"p1": tune.choice([0]),
                "p2": tune.choice([0]),
                "p3": tune.choice([0]),
                "p4": tune.choice([0]),
                "p5": tune.choice([0])},
        event_queue=Queue(),
        event_creator=Event,
        # callbacks=[JobEvent(app, queue)],
        scheduler_trial_runner_queue={"downlink": Queue(), "uplink": Queue()},
        inactivity_time=30)



if __name__ == '__main__':



    ray.init(address="10.1.1.2:6379", _redis_password="tf_cluster_123")

    total_capacity = ray.cluster_resources()["GPU"]
    available_capacity = ray.available_resources()["GPU"]


    future = tune_cifar10.remote()
    result = ray.get(future, timeout=200.0)