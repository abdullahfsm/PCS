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

        print(f"avail_resources: {self._avail_resources.gpu}")
        print(f"committed_resources: {self._committed_resources.gpu}")

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


        if trial.uses_placement_groups:
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
                    print(f"""DEBUG: cond(1),
                            self._wait_for_pg: {self._wait_for_pg}
                            just_staged: {just_staged}
                            self._trial_just_finished_before: {self._trial_just_finished_before}
                            self.get_running_trials(): {self.get_running_trials()}
                            """)
                    # print(f"DEBUG: cond (1)")

                    return None

            if not self._pg_manager.has_ready(trial):
                # PG may have become ready during waiting period

                print(f"DEBUG: cond (2)")
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



    def start_trial(self,
                    trial: Trial,
                    checkpoint: Optional[Checkpoint] = None,
                    train: bool = True) -> bool:

        
        has_resources =  self.has_resources_for_trial(trial)

        if has_resources:
            self._commit_resources(trial.resources)

            print(f"committing resource to trial: {trial.trial_id}")

            
            start_val = self._start_trial(trial, checkpoint, train=train)
            # start_val = super(MyRayTrialExecutor, self).start_trial(trial, checkpoint, train)
            print(f"start_val: {start_val}")

            return start_val
        return False


    def stop_trial(self,
                   trial: Trial,
                   error: bool = False,
                   error_msg: Optional[str] = None,
                   destroy_pg_if_cannot_replace: bool = True) -> None:

        super(MyRayTrialExecutor, self).stop_trial(trial, error, error_msg, destroy_pg_if_cannot_replace)
        self._return_resources(trial.resources)



    # def _stop_trial(self,
    #                 trial: Trial,
    #                 error=False,
    #                 error_msg=None,
    #                 destroy_pg_if_cannot_replace=True):
    #     """Stops this trial.

    #     Stops this trial, releasing all allocating resources. If stopping the
    #     trial fails, the run will be marked as terminated in error, but no
    #     exception will be thrown.

    #     If the placement group will be used right away
    #     (destroy_pg_if_cannot_replace=False), we do not remove its placement
    #     group (or a surrogate placement group).

    #     Args:
    #         error (bool): Whether to mark this trial as terminated in error.
    #         error_msg (str): Optional error message.

    #     """
    #     self.set_status(trial, Trial.ERROR if error else Trial.TERMINATED)
    #     self._trial_just_finished = True
    #     trial.set_location(Location())

    #     try:
    #         trial.write_error_log(error_msg)
    #         if hasattr(trial, "runner") and trial.runner:
    #             if (not error and self._reuse_actors
    #                     and (len(self._cached_actor_pg) <
    #                          (self._cached_actor_pg.maxlen or float("inf")))):
    #                 logger.debug("Reusing actor for %s", trial.runner)
    #                 # Move PG into cache (disassociate from trial)
    #                 pg = self._pg_manager.cache_trial_pg(trial)
    #                 if pg or not trial.uses_placement_groups:
    #                     # True if a placement group was replaced
    #                     self._cached_actor_pg.append((trial.runner, pg))
    #                     should_destroy_actor = False
    #                 else:
    #                     # False if no placement group was replaced. This should
    #                     # only be the case if there are no more trials with
    #                     # this placement group factory to run
    #                     logger.debug(
    #                         "Could not cache of trial {trial} actor for "
    #                         "reuse, as there are no pending trials "
    #                         "requiring its resources.")
    #                     should_destroy_actor = True
    #             else:
    #                 should_destroy_actor = True

    #             if should_destroy_actor:
    #                 logger.debug("Trial %s: Destroying actor.", trial)

    #                 # Try to return the placement group for other trials to use

    #                 try:

    #                     self._pg_manager.return_pg(trial,
    #                                                destroy_pg_if_cannot_replace)
    #                 except Exception as e:
    #                     print(f"WarningError: {e}")

    #                 with self._change_working_directory(trial):
    #                     self._trial_cleanup.add(trial, actor=trial.runner)

    #             if trial in self._staged_trials:
    #                 self._staged_trials.remove(trial)
    #             else:
    #                 print(f"trial wasn't staged")

    #     except Exception:
    #         logger.exception("Trial %s: Error stopping runner.", trial)
    #         self.set_status(trial, Trial.ERROR)
    #     finally:
    #         trial.set_runner(None)


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

