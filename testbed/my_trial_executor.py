# coding: utf-8
import copy
from collections import deque
from functools import partial
import logging
import os
import random
import time
import traceback
from contextlib import contextmanager
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
)

import ray
from ray.actor import ActorHandle
from ray.exceptions import GetTimeoutError
from ray import ray_constants
from ray._private.resource_spec import ResourceSpec, NODE_ID_PREFIX
from ray.tune.durable_trainable import DurableTrainable
from ray.tune.error import AbortTrialExecution, TuneError
from ray.tune.logger import NoopLogger
from ray.tune.result import TRIAL_INFO, STDOUT_FILE, STDERR_FILE
from ray.tune.resources import Resources
from ray.tune.utils.placement_groups import PlacementGroupManager, \
    get_tune_pg_prefix
from ray.tune.utils.trainable import TrainableUtil
from ray.tune.trial import Trial, Checkpoint, Location, TrialInfo
from ray.tune.trial_executor import TrialExecutor
from ray.tune.utils import warn_if_slow
from ray.util import log_once
from ray.util.annotations import DeveloperAPI


from ray.tune.schedulers import TrialScheduler

from ray.tune.ray_trial_executor import *
from ray.util.queue import Queue, Empty
from ray.tune.ray_trial_executor import (
    _ActorClassCache,
    _class_cache,
    _LocalWrapper,
    _TrialCleanup,
    _to_gb
)

from common import Event
import datetime

class MyRayTrialExecutor(RayTrialExecutor):
    """An implementation of MyRayTrialExecutor based on RayTrialExecutor."""

    def __init__(self,
                name,
                get_queue,
                set_queue,
                event_queue,
                init_resources: Resources = Resources(cpu=0,gpu=0),
                queue_trials: bool = False,
                reuse_actors: bool = False,
                result_buffer_length: Optional[int] = None,
                refresh_period: Optional[float] = None,
                wait_for_placement_group: Optional[float] = None,
                ):
        
        
        self._name = name
        self._get_queue = get_queue
        self._set_queue = set_queue
        self._event_queue = event_queue

        self._tune_started = False


        super(MyRayTrialExecutor, self).__init__(queue_trials, reuse_actors, result_buffer_length, refresh_period, wait_for_placement_group)

        self._avail_resources = init_resources

        self._demand = Resources(cpu=0,gpu=0)


    def _get_app_id(self):
        try:
            return int(self._name.split("app_")[-1])
        except Exception as e:
            raise e


    def _get_job_id(self, trial):
        try:
            return int(trial.trial_id.split("_")[-1])
        except Exception as e:
            raise e

    def _notify_tune_finished(self):
        app_id = self._get_app_id()
        self._event_queue.put(Event(event_id=app_id, event_type=Event.APP_END, event_time=datetime.datetime.now(), app_id=app_id))

    def _notify_tune_start(self):
        app_id = self._get_app_id()
        self._event_queue.put(Event(event_id=app_id, event_type=Event.APP_START, event_time=datetime.datetime.now(), app_id=app_id))

    def _notify_trial_start(self, trial):
        app_id = self._get_app_id()
        job_id = self._get_job_id(trial)
        self._event_queue.put(Event(event_id=app_id, event_type=Event.JOB_START, event_time=datetime.datetime.now(), app_id=app_id, job_id=job_id))

    def _notify_trial_end(self, trial):
        app_id = self._get_app_id()
        job_id = self._get_job_id(trial)
        self._event_queue.put(Event(event_id=app_id, event_type=Event.JOB_END, event_time=datetime.datetime.now(), app_id=app_id, job_id=job_id))


    def debug_string(self) -> str:
        """Returns a human readable message for printing to the console."""

        

        if self._resources_initialized:
            status = ("Resources used: {}/{} CPUs, {}/{} GPUs, "
                      "{}/{} GiB heap, {}/{} GiB objects, GPU demand: {}/{}".format(
                        self._committed_resources.cpu, self._avail_resources.cpu,
                        self._committed_resources.gpu,self._avail_resources.gpu,
                        _to_gb(self._committed_resources.memory),_to_gb(self._avail_resources.memory),
                        _to_gb(self._committed_resources.object_store_memory),_to_gb(self._avail_resources.object_store_memory),
                        self._demand.gpu, self._avail_resources.gpu,)
                      )
            return status
        else:
            return "Resources requested: ?"


    def _update_demand(self, trials: List[Trial]):
        self._demand = Resources(cpu=0,gpu=0)
        
        for trial in trials:
            self._demand += trial.resources
            
        

    def on_step_end(self, trial_runner) -> None:
        

        trials = trial_runner.get_trials()

        if trial_runner.is_finished():
            self._notify_tune_finished()

        self._just_staged_trials.clear()

        if time.time() > self.last_pg_recon + self.pg_recon_interval:
            # Only do this every now and then - usually the placement groups
            # should not get out of sync, and calling this often is inefficient
            self._pg_manager.reconcile_placement_groups(trials)
            self.last_pg_recon = time.time()

        self._pg_manager.cleanup()





    def on_step_begin(self, trial_runner) -> None:
        """Before step() is called, update the available resources."""

        trials = trial_runner.get_trials()

        if not self._tune_started:
            self._notify_tune_start()
            self._tune_started = True



        self._update_avail_resources()
        self._update_demand(trials)



        # TODO: define preempt, unpreempt state, define preempt and unpreempt functions
        if self._avail_resources >= self._demand:
            if self._demand > self._committed_resources:
                # start to unpause/unpreempt d-c trials
                pass
            elif self._demand == self._committed_resources:
                # do nothing
                pass
            else:
                raise ValueError("Demand is less than committed resources")
        else:
            if self._avail_resources > self._committed_resources:
                # unpause unpreempt r-c trials
                pass
            elif self._avail_resources == self._committed_resources:
                # do nothing
                pass
            else:
                # only point where preemption happens
                # preempt c-r trials
                running_trials = list(filter(lambda t: t.status == Trial.RUNNING, trials))

                get_trial_idx = lambda t: int(t.trial_id.split('_')[-1])

                running_trials = sorted(running_trials, key=get_trial_idx, reverse=True)

                to_preempt = self._committed_resources.gpu - self._avail_resources.gpu

                assert(to_preempt <= len(running_trials))

                # preempt the last r-c trials
                for t in range(to_preempt):
                    trial_runner._queue_decision(running_trials[t], TrialScheduler.PAUSE)
            
        self._trial_just_finished_before = self._trial_just_finished
        self._trial_just_finished = False


    def has_resources(self, resources: Resources) -> bool:
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

    def has_resources_for_trial(self, trial: Trial) -> bool:
        """Returns whether this runner has resources available for this trial.

        Args:
            trial: Trial object which should be scheduled.

        Returns:
            boolean

        """
        return self.has_resources(trial.resources)



    def start_trial(self,
                    trial: Trial,
                    checkpoint: Optional[Checkpoint] = None,
                    train: bool = True) -> bool:

        
        has_resources =  self.has_resources_for_trial(trial)

        if has_resources:

            prior_status = trial.status

            start_val = super(MyRayTrialExecutor, self).start_trial(trial, checkpoint, train)

            if start_val:
                # print(f"committing resource to trial: {trial.trial_id}")
                self._commit_resources(trial.resources)
                
                if prior_status == Trial.PENDING:
                    self._notify_trial_start(trial)

            return start_val
        
        return False


    def stop_trial(self,
                   trial: Trial,
                   error: bool = False,
                   error_msg: Optional[str] = None,
                   destroy_pg_if_cannot_replace: bool = True) -> None:

        super(MyRayTrialExecutor, self).stop_trial(trial, error, error_msg, destroy_pg_if_cannot_replace)
        self._return_resources(trial.resources)


        print(f"Stopping trail_id: {trial.trial_id} status: {trial.status}")

        if not self._find_item(self._paused, trial):
            # stop_trial was not called by pause_trial
            self._notify_trial_end(trial)



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

        else:
            raise ValueError("self._get_queue is not initialized")
