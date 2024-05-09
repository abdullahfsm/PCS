from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Union,
    Any,
)

import ray
from ray import tune
import numpy as np
import argparse



from ray.tune.schedulers.timed_fifo import TimedFIFOScheduler as TimedFIFO
from ray.tune.utils.placement_groups import PlacementGroupFactory


from ray.tune.trial import Trial
from ray.tune.schedulers import ResourceChangingScheduler, ASHAScheduler
from ray.tune import Trainable
from ray.tune.resources import Resources

from tensorflow.keras import datasets, layers, models, regularizers, Input
from ray.tune.integration.keras import TuneReportCallback
from ray.tune.integration.keras import TuneReportCheckpointCallback


from ray.tune.ray_trial_executor import RayTrialExecutor

from filelock import FileLock
import os


from ray.util.queue import Queue, Empty

from common import Event

import time



CHECKPOINT_FILENAME="my_model.keras"



class MyRayTrialExecutor(RayTrialExecutor):
    """An implementation of MyRayTrialExecutor based on RayTrialExecutor."""

    def __init__(self,
                get_queue,
                set_queue,
                queue_trials: bool = False,
                reuse_actors: bool = False,
                result_buffer_length: Optional[int] = None,
                refresh_period: Optional[float] = None,
                wait_for_placement_group: Optional[float] = None,
                init_resources: Resources = Resources(cpu=0,gpu=0)):
        

        
        self._set_queue = set_queue
        self._get_queue = get_queue
        super(MyRayTrialExecutor, self).__init__(queue_trials, reuse_actors, result_buffer_length, refresh_period, wait_for_placement_group)

        print(f"get_queue: {self._get_queue}")


        self._avail_resources = init_resources


    def _update_avail_resources(self, num_retries=5):

        if self._get_queue is not None:
            try:
                resources = self._get_queue.get(block=False)
                if not isinstance(resources, Resources):
                    raise ValueError(f"resources not of type Resources")
                print(f"Got resources: {resources}")
            except Empty:
                # do nothing. no need to update
                pass
            except Exception as e:
                raise e


            self._last_resource_refresh = time.time()
            self._resources_initialized = True
            # self._avail_resources = Resources(
            #     int(num_cpus),
            #     int(num_gpus),
            #     memory=int(memory),
            #     object_store_memory=int(object_store_memory),
            #     custom_resources=custom_resources)
            # self._last_resource_refresh = time.time()            
        else:
            raise ValueError("self._get_queue is not initialized")
        # else:
        #     if ray.is_initialized():
        #         super(MyRayTrialExecutor, self)._update_avail_resources()

class App(object):
    """docstring for App"""
    def __init__(self, app_id, service, demand):
        super(App, self).__init__()
        self.app_id = app_id
        self.service = service
        self.demand = demand


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



def train_cifar10(config: dict, checkpoint_dir=None):

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

    # load weights
    if checkpoint_dir:
        model = tf.keras.models.load_model(os.path.join(checkpoint_dir, CHECKPOINT_FILENAME))


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
            callbacks=[
                TuneReportCheckpointCallback(
                    filename=CHECKPOINT_FILENAME,
                    # checkpointing should happen every iteration
                    # with dynamic resource allocation
                    frequency=1)


            ])


@ray.remote
def schedule_q_put(sleep_time, queue, item):
    sleep(sleep_time)
    queue.put(item)


def example_resources_allocation_function(
        trial_runner: "trial_runner.TrialRunner", trial: Trial,
        result: Dict[str, Any], scheduler: "ResourceChangingScheduler"
) -> Union[None, PlacementGroupFactory, Resources]:
    base_trial_resource = scheduler._base_trial_resources or PlacementGroupFactory([{"CPU": 1, "GPU": 1}])
    min_gpu = base_trial_resource.required_resources.get("GPU", 1)

    # Get the number of GPUs available in total (not just free)
    total_available_gpus = (trial_runner.trial_executor._avail_resources.gpu)

    get_trial_idx = lambda t: int(t.trial_id.split('_')[-1])


    all_trials = trial_runner.get_live_trials()
    trial_idxs = [get_trial_idx(t) for t in all_trials]

    gpu_allocation = [0] * len(all_trials)

    per_trial_gpu_allocation = 1

    print(f"DEBUG: {trial_idxs}")

    for trial_idx in len(trial_idxs):
        allocation = min(per_trial_gpu_allocation, total_available_gpus)
        gpu_allocation[trial_idx] = allocation
        total_available_gpus -= allocation


    # trial_runner.trial_executor.resources
    print("++++++++++++++++++++++")
    print(f"Trial_id: {trial.trial_id} has resources: {trial.resources} and placement group: {trial.placement_group_factory} and total gpus are: {total_available_gpus} and has been assigned {gpu_allocation[get_trial_idx(trial)]}")
    print("++++++++++++++++++++++")


    # # Divide the free CPUs among all live trials
    # cpu_to_use = max(
    #     min_cpu,
    #     total_available_cpus // len(trial_runner.get_live_trials()))
    # Assign new CPUs to the trial in a PlacementGroupFactory
    return PlacementGroupFactory([{"CPU": 1 if get_trial_idx(trial) > 0 else 0, "GPU": get_trial_idx(trial)}])
    # return PlacementGroupFactory([{"CPU": 1, "GPU": total_available_gpus//len(trial_runner.get_live_trials())}])








def tune_cifar10(num_samples=2, reduction_factor=2, budget=10.0):

    app = App(0, budget, num_samples)

    trial_scheduler=TimedFIFO(time_attr='time_total_s',budget=(app.service/app.demand))

    trial_scheduler = ResourceChangingScheduler(
        base_scheduler=trial_scheduler,
        resources_allocation_function=example_resources_allocation_function
        )


    queue = Queue()
    queue.put(Resources(cpu=1,gpu=1))


    trial_executor = MyRayTrialExecutor(get_queue=queue, set_queue=Queue, init_resources=Resources(cpu=1,gpu=1))

    analysis = tune.run(
        train_cifar10,
        resources_per_trial=PlacementGroupFactory([{"CPU": 1, "GPU": 1}]),
        name=f"app_{app.app_id}",
        trial_name_creator=lambda T: "app_%d_%s" % (app.app_id, T.trial_id),
        scheduler=trial_scheduler,
        num_samples=num_samples,
        config={"p1": tune.choice([0,1]),
                "p2": tune.choice([0,1]),
                "p3": tune.choice([0,1]),
                "p4": tune.choice([0,1]),
                "p5": tune.choice([0,1])},
        
        trial_executor=trial_executor,
    )
        


    print("Best config: ", analysis.get_best_config(
        metric="accuracy", mode="max"))

    # Get a dataframe for analyzing trial results.
    # df = analysis.results_df


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=14)
    parser.add_argument("--reduction_factor", type=int, default=2)
    parser.add_argument("--budget", type=float, default=5.0)
    args = parser.parse_args()


    os.environ["TUNE_CLUSTER_SSH_KEY"] = f"{os.path.expanduser('~')}/.ssh/key"

    ray.init(address="auto")
    tune_cifar10(num_samples=args.num_samples, reduction_factor=args.reduction_factor, budget=args.budget)

    time.sleep(2)
