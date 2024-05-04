import logging
from typing import Dict, Optional, Union
import sys

import numpy as np
import pickle

from ray.tune.schedulers.trial_scheduler import FIFOScheduler
from ray.tune.experiment import Trial
from ray.tune.execution.tune_controller import TuneController


from datetime import datetime, timedelta


logger = logging.getLogger(__name__)


class TimedFIFOScheduler(FIFOScheduler):
    """Simple scheduler that just runs trials in submission order."""

    def __init__(self,
                time_attr: str = "training_iteration",
                budget: float = 100.0):

        self._budget = budget
        self._time_attr = time_attr

        self._trial_id_to_time_elapsed = {}
        self._trial_id_to_last_event_time = {}
        self._trial_id_to_budget = {}


    def on_trial_pause(self, tune_controller: "TuneController", trial: Trial):

        self._trial_id_to_last_event_time[trial.trial_id] = None



    def on_trial_unpause(self, tune_controller: "TuneController", trial: Trial):

        self._trial_id_to_last_event_time[trial.trial_id] = datetime.now()


    def on_trial_add(self, tune_controller: "TuneController", trial: Trial):
        self._trial_id_to_time_elapsed[trial.trial_id] = 0
        self._trial_id_to_last_event_time[trial.trial_id] = datetime.now()
        self._trial_id_to_budget[trial.trial_id] = self._budget


    def on_trial_result(
        self, tune_controller: "TuneController", trial: Trial, result: Dict
    ) -> str:
    
        if self._trial_id_to_last_event_time[trial.trial_id] != None:
            now_time = datetime.now()
            self._trial_id_to_time_elapsed[trial.trial_id] += (now_time - self._trial_id_to_last_event_time[trial.trial_id]).total_seconds()
            self._trial_id_to_last_event_time[trial.trial_id] = now_time

        if self._trial_id_to_time_elapsed[trial.trial_id] <= self._budget:
            return FIFOScheduler.CONTINUE
        else:
            return FIFOScheduler.STOP

    def debug_string(self) -> str:
        return "Using TimedFIFO scheduling algorithm."

    def estimate_remaining_trial_times(self):
        trial_id_to_estimated_remaining_time = {}
        for trial_id in self._trial_id_to_time_elapsed:
            trial_id_to_estimated_remaining_time[trial_id] = max(0, self._trial_id_to_budget[trial_id] - self._trial_id_to_time_elapsed[trial_id])
        return trial_id_to_estimated_remaining_time

if __name__ == '__main__':
    trial_scheduler=TimedFIFOScheduler(time_attr='time_total_s',budget=(1000/5))



