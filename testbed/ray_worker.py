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


from ray.tune.trial import Trial, Checkpoint, Location, TrialInfo
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

        self._avail_resources = init_resources

        print(f"avail_resources: {self._avail_resources.gpu}")
        print(f"committed_resources: {self._committed_resources.gpu}")

    def _has_resources(self, resources: Resources) -> bool:
        """Returns whether this runner has at least the specified resources.

        This refreshes the Ray cluster resources if the time since last update
        has exceeded self._refresh_period. This also assumes that the
        cluster is not resizing very frequently.
        """

        self._update_avail_resources()
        currently_available = Resources.subtract(self._avail_resources,
                                                 self._committed_resources)
        have_space = (
            resources.cpu_total() <= currently_available.cpu
            and resources.gpu_total() <= currently_available.gpu
            and resources.memory_total() <= currently_available.memory
            and resources.object_store_memory_total() <=
            currently_available.object_store_memory and all(
                resources.get_res_total(res) <= currently_available.get(res)
                for res in resources.custom_resources))

        if have_space:
            # The assumption right now is that we block all trials if one
            # trial is queued.
            self._trial_queued = False
            return True

        return False

    def start_trial(self,
                    trial: Trial,
                    checkpoint: Optional[Checkpoint] = None,
                    train: bool = True) -> bool:

        
        has_resources =  self._has_resources(trial.resources)

        # print(f"trial_id: {trial.trial_id}, _avail_resources: {self._avail_resources.gpu}, committed_resources: {self._committed_resources.gpu}, requested: {trial.resources.gpu}, has_resources: {has_resources}")

        if has_resources:
            self._commit_resources(trial.resources)

            print(f"committing resource to trial: {trial.trial_id}")

            
            start_val = self._start_trial(trial, checkpoint, train=train)
            # start_val = super(MyRayTrialExecutor, self).start_trial(trial, checkpoint, train)
            print(f"start_val: {start_val}")

            return start_val
        return False


    def _setup_remote_runner(self, trial):
        trial.init_logdir()
        # We checkpoint metadata here to try mitigating logdir duplication
        self.try_checkpoint_metadata(trial)
        logger_creator = partial(noop_logger_creator, logdir=trial.logdir)

        if self._reuse_actors and len(self._cached_actor_pg) > 0:

            raise ValueError("Not to resuse Actors")


            existing_runner, pg = self._cached_actor_pg.popleft()
            logger.debug(f"Trial {trial}: Reusing cached runner "
                         f"{existing_runner}")

            trial.set_runner(existing_runner)
            if pg and trial.uses_placement_groups:
                self._pg_manager.assign_cached_pg(pg, trial)

            if not self.reset_trial(trial, trial.config, trial.experiment_tag,
                                    logger_creator):
                raise AbortTrialExecution(
                    "Trainable runner reuse requires reset_config() to be "
                    "implemented and return True.")
            return existing_runner

        if len(self._cached_actor_pg) > 0:
            existing_runner, pg = self._cached_actor_pg.popleft()

            logger.debug(
                f"Cannot reuse cached runner {existing_runner} for new trial")

            if pg:
                self._pg_manager.return_or_clean_cached_pg(pg)

            with self._change_working_directory(trial):
                self._trial_cleanup.add(trial, actor=existing_runner)

        trainable_cls = trial.get_trainable_cls()
        if not trainable_cls:
            raise AbortTrialExecution(
                f"Invalid trainable: {trial.trainable_name}. If you passed "
                f"a string, make sure the trainable was registered before.")
        _actor_cls = _class_cache.get(trainable_cls)


        # if trial.uses_placement_groups:
        if False:
            if not self._pg_manager.has_ready(trial, update=True):
                if trial not in self._staged_trials:
                    if self._pg_manager.stage_trial_pg(trial):
                        self._staged_trials.add(trial)
                        self._just_staged_trials.add(trial)

                just_staged = trial in self._just_staged_trials

                # This part of the code is mostly here for testing
                # purposes. If self._wait_for_pg is set, we will wait here
                # for that many seconds until the placement group is ready.
                # This ensures that the trial can be started right away and
                # not just in the next step() of the trial runner.
                # We only do this if we have reason to believe that resources
                # will be ready, soon, i.e. when a) we just staged the PG,
                # b) another trial just exited, freeing resources, or c)
                # when there are no currently running trials.
                if self._wait_for_pg is not None and (
                        just_staged or self._trial_just_finished_before
                        or not self.get_running_trials()):
                    logger.debug(
                        f"Waiting up to {self._wait_for_pg} seconds for "
                        f"placement group of trial {trial} to become ready.")
                    wait_end = time.monotonic() + self._wait_for_pg
                    while time.monotonic() < wait_end:
                        self._pg_manager.update_status()
                        if self._pg_manager.has_ready(trial):
                            break
                        time.sleep(0.1)
                else:
                    return None

            if not self._pg_manager.has_ready(trial):
                # PG may have become ready during waiting period
                return None

            full_actor_class = self._pg_manager.get_full_actor_cls(
                trial, _actor_cls)
        else:
            full_actor_class = _actor_cls.options(
                num_cpus=trial.resources.cpu,
                num_gpus=trial.resources.gpu,
                memory=trial.resources.memory or None,
                object_store_memory=trial.resources.object_store_memory
                or None,
                resources=trial.resources.custom_resources)
        # Clear the Trial's location (to be updated later on result)
        # since we don't know where the remote runner is placed.
        trial.set_location(Location())
        logger.debug("Trial %s: Setting up new remote runner.", trial)
        # Logging for trials is handled centrally by TrialRunner, so
        # configure the remote runner to use a noop-logger.
        trial_config = copy.deepcopy(trial.config)
        trial_config[TRIAL_INFO] = TrialInfo(trial)

        stdout_file, stderr_file = trial.log_to_file
        trial_config[STDOUT_FILE] = stdout_file
        trial_config[STDERR_FILE] = stderr_file
        kwargs = {
            "config": trial_config,
            "logger_creator": logger_creator,
        }
        if issubclass(trial.get_trainable_cls(), DurableTrainable):
            kwargs["remote_checkpoint_dir"] = trial.remote_checkpoint_dir
            kwargs["sync_function_tpl"] = trial.sync_to_cloud

        print(f"DEBUG: full_actor_class: {full_actor_class}")

        with self._change_working_directory(trial):
            return full_actor_class.remote(**kwargs)



    def _start_trial(self, trial, checkpoint=None, runner=None,
                     train=True) -> bool:
        """Starts trial and restores last result if trial was paused.

        Args:
            trial (Trial): The trial to start.
            checkpoint (Optional[Checkpoint]): The checkpoint to restore from.
                If None, and no trial checkpoint exists, the trial is started
                from the beginning.
            runner (Trainable): The remote runner to use. This can be the
                cached actor. If None, a new runner is created.
            train (bool): Whether or not to start training.

        Returns:
            True if trial was started successfully, False otherwise.

        See `RayTrialExecutor.restore` for possible errors raised.
        """
        prior_status = trial.status
        self.set_status(trial, Trial.PENDING)
        if runner is None:
            runner = self._setup_remote_runner(trial)
            if not runner:
                print("unable to set runner!")
                return False
        trial.set_runner(runner)
        self._notify_trainable_of_new_resources_if_needed(trial)
        self.restore(trial, checkpoint)
        self.set_status(trial, Trial.RUNNING)

        if trial in self._staged_trials:
            self._staged_trials.remove(trial)

        previous_run = self._find_item(self._paused, trial)
        if prior_status == Trial.PAUSED and previous_run:
            # If Trial was in flight when paused, self._paused stores result.
            self._paused.pop(previous_run[0])
            self._running[previous_run[0]] = trial
        elif train and not trial.is_restoring:
            self._train(trial)
        return True




    def stop_trial(self,
                   trial: Trial,
                   error: bool = False,
                   error_msg: Optional[str] = None,
                   destroy_pg_if_cannot_replace: bool = True) -> None:

        super(MyRayTrialExecutor, self).stop_trial(trial, error, error_msg, destroy_pg_if_cannot_replace)
        self._return_resources(trial.resources)



    def _update_avail_resources(self, num_retries=5):

        if self._get_queue is not None:
            try:
                resources = self._get_queue.get(block=False)
                if not isinstance(resources, Resources):
                    raise ValueError(f"resources not of type Resources")
                print(f"Got resources: {resources}")

                self._avail_resources = resources

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

    gpu_allocations = [0] * len(all_trials)

    per_trial_gpu_allocations = 1

    print(f"DEBUG: {trial_idxs}")

    for trial_idx in trial_idxs:
        allocation = min(per_trial_gpu_allocations, total_available_gpus)
        gpu_allocations[trial_idx] = allocation
        total_available_gpus -= allocation


    gpu_allocation = gpu_allocations[get_trial_idx(trial)]

    # trial_runner.trial_executor.resources
    print("++++++++++++++++++++++")
    print(f"Trial_id: {trial.trial_id} has resources: {trial.resources} and placement group: {trial.placement_group_factory} and total gpus are: {trial_runner.trial_executor._avail_resources.gpu} and has been assigned {gpu_allocation}")
    print("++++++++++++++++++++++")


    # # Divide the free CPUs among all live trials
    # cpu_to_use = max(
    #     min_cpu,
    #     total_available_cpus // len(trial_runner.get_live_trials()))
    # Assign new CPUs to the trial in a PlacementGroupFactory



    return PlacementGroupFactory([{"CPU": 1 if gpu_allocation > 0 else 0, "GPU": gpu_allocation}])
    # return PlacementGroupFactory([{"CPU": 1, "GPU": total_available_gpus//len(trial_runner.get_live_trials())}])



def tune_cifar10(num_samples=2, reduction_factor=2, budget=10.0):

    app = App(0, budget, num_samples)

    trial_scheduler=TimedFIFO(time_attr='time_total_s',budget=(app.service/app.demand))

    # trial_scheduler = ResourceChangingScheduler(
    #     base_scheduler=trial_scheduler,
    #     resources_allocation_function=example_resources_allocation_function
    #     )


    queue = Queue()
    queue.put(Resources(cpu=1,gpu=1))


    trial_executor = MyRayTrialExecutor(get_queue=queue, set_queue=Queue, init_resources=Resources(cpu=1,gpu=1))

    analysis = tune.run(
        train_cifar10,
        # resources_per_trial=PlacementGroupFactory([{"CPU": 1, "GPU": 1}]),
        resources_per_trial={'cpu': 1, 'gpu': 1},
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
