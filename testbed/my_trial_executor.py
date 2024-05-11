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

        self._preempted = {}
        self._pending = {}

        self._demand = Resources(cpu=0,gpu=0)


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
        
        # for lst in self._running.values() + self._pending.values() + self._preempted.values() + self._paused.values():
            
        

    def on_step_begin(self, trial_runner) -> None:
        """Before step() is called, update the available resources."""

        trials = trial_runner.get_trials()


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
                running_trials = list(filter(lambda t: t.status == Trial.RUNNING))

                get_trial_idx = lambda t: int(t.trial_id.split('_')[-1])

                running_trials = sorted(running_trials, key=get_trial_idx, reverse=True)

                to_preempt = self._committed_resources.gpu - self._avail_resources.gpu

                assert(to_preempt <= len(running_trials))

                for r in range(to_preempt):
                    trial_runner._queue_decision(running_trials[r], TrialScheduler.PAUSE)
            
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

            start_val = super(MyRayTrialExecutor, self).start_trial(trial, checkpoint, train)

            if start_val:
                print(f"committing resource to trial: {trial.trial_id}")
                self._commit_resources(trial.resources)
                
                if trial.trial_id in self._pending:
                    self._pending.pop(trial.trial_id)

            return start_val
        self._pending[trial.trial_id] = trial
        return False


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

        else:
            raise ValueError("self._get_queue is not initialized")
