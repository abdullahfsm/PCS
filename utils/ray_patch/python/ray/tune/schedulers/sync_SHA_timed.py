import logging
from typing import Dict, Optional, Union
import sys

import numpy as np
import pickle

from ray.tune import trial_runner
from ray.tune.result import DEFAULT_METRIC
from ray.tune.schedulers.trial_scheduler import FIFOScheduler, TrialScheduler
from ray.tune.trial import Trial

from datetime import datetime, timedelta


logger = logging.getLogger(__name__)


class SyncSHATimedScheduler(FIFOScheduler):
    """Simple SHA that gives different trials different running budget."""

    def __init__(self,
                time_attr: str = "training_iteration",
                budget: float = 100.0,
                num_samples = 2,
                reduction_factor=2):

        self._budget = budget
        self._time_attr = time_attr

        self._trial_id_to_budget = {}
        self._budget_per_stage = list()

        self._trial_id_to_time_elapsed = {}
        self._trial_id_to_last_event_time = {}


        self._num_samples = num_samples
        self._reduction_factor = reduction_factor

        self._total_stages = int(np.floor(np.log(self._num_samples)/np.log(self._reduction_factor))) + 1

        
        active_trials = [int(np.floor(self._num_samples/np.power(self._reduction_factor, stage))) for stage in range(self._total_stages)]
        trials_per_stage = [active_trials[stage-1] - active_trials[stage] for stage in range(1,self._total_stages)] + [1]
        fraction_per_stage = [np.power(self._reduction_factor, stage) for stage in range(self._total_stages)]
        tau = self._budget/np.dot(trials_per_stage, fraction_per_stage)

        self._samples_added = 0

        '''
        print(np.dot(jobs_per_stage, fraction_per_stage))
        print(jobs_per_stage)
        print(fraction_per_stage)
        print("=================")
        '''

        # exponentially increasing budget

        service_per_stage = np.multiply(tau, fraction_per_stage)
        trials_per_stage = list(np.cumsum(trials_per_stage))

        stage = 0
        for i in range(self._num_samples):

            if i == trials_per_stage[stage]:
                stage += 1

            self._budget_per_stage.append(service_per_stage[stage])




    def on_trial_pause(self, trial_runner: "trial_runner.TrialRunner",
                          trial: Trial):

        self._trial_id_to_last_event_time[trial.trial_id] = None



    def on_trial_unpause(self, trial_runner: "trial_runner.TrialRunner",
                          trial: Trial):

        self._trial_id_to_last_event_time[trial.trial_id] = datetime.now()


    def on_trial_add(self, trial_runner: "trial_runner.TrialRunner",
                     trial: Trial):
        self._trial_id_to_time_elapsed[trial.trial_id] = 0
        self._trial_id_to_last_event_time[trial.trial_id] = datetime.now()
        self._trial_id_to_budget[trial.trial_id] = self._budget_per_stage.pop(0)

        self._samples_added += 1
        assert(self._samples_added <= self._num_samples)



    def on_trial_result(self, trial_runner: "trial_runner.TrialRunner",
                        trial: Trial, result: Dict) -> str:

        
        if self._trial_id_to_last_event_time[trial.trial_id] != None:
            now_time = datetime.now()
            self._trial_id_to_time_elapsed[trial.trial_id] += (now_time - self._trial_id_to_last_event_time[trial.trial_id]).total_seconds()
            self._trial_id_to_last_event_time[trial.trial_id] = now_time


        if self._trial_id_to_time_elapsed[trial.trial_id] <= self._trial_id_to_budget[trial.trial_id]:
            action = TrialScheduler.CONTINUE
        else:
            action = TrialScheduler.STOP

        return action

    def debug_string(self) -> str:
        return "Using SyncSHATimed scheduling algorithm."


    def estimate_remaining_trial_times(self):
        trial_id_to_estimated_remaining_time = {}
        for trial_id in self._trial_id_to_time_elapsed:
            trial_id_to_estimated_remaining_time[trial_id] = max(0, self._trial_id_to_budget[trial_id] - self._trial_id_to_time_elapsed[trial_id])
        return trial_id_to_estimated_remaining_time