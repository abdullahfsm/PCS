from typing import Dict, Optional, TYPE_CHECKING

from ray.air._internal.usage import tag_scheduler
from ray.tune.result import DEFAULT_METRIC
from ray.tune.experiment import Trial
from ray.util.annotations import DeveloperAPI, PublicAPI

if TYPE_CHECKING:
    from ray.tune.execution.tune_controller import TuneController


@DeveloperAPI
class TrialScheduler:
    """Interface for implementing a Trial Scheduler class.

    Note to Tune developers: If a new scheduler is added, please update
    `air/_internal/usage.py`.
    """

    CONTINUE = "CONTINUE"  #: Status for continuing trial execution
    PAUSE = "PAUSE"  #: Status for pausing trial execution
    STOP = "STOP"  #: Status for stopping trial execution
    # Caution: Temporary and anti-pattern! This means Scheduler calls
    # into Executor directly without going through TrialRunner.
    # TODO(xwjiang): Deprecate this after we control the interaction
    #  between schedulers and executor.
    NOOP = "NOOP"

    _metric = None

    _supports_buffered_results = True

    def __init__(self):
        tag_scheduler(self)

    @property
    def metric(self):
        return self._metric

    @property
    def supports_buffered_results(self):
        return self._supports_buffered_results

    def set_search_properties(
        self, metric: Optional[str], mode: Optional[str], **spec
    ) -> bool:
        """Pass search properties to scheduler.

        This method acts as an alternative to instantiating schedulers
        that react to metrics with their own `metric` and `mode` parameters.

        Args:
            metric: Metric to optimize
            mode: One of ["min", "max"]. Direction to optimize.
            **spec: Any kwargs for forward compatiblity.
                Info like Experiment.PUBLIC_KEYS is provided through here.
        """
        if self._metric and metric:
            return False
        if metric:
            self._metric = metric

        if self._metric is None:
            # Per default, use anonymous metric
            self._metric = DEFAULT_METRIC

        return True

    def on_trial_add(self, tune_controller: "TuneController", trial: Trial):
        """Called when a new trial is added to the trial runner."""

        raise NotImplementedError

    def on_trial_error(self, tune_controller: "TuneController", trial: Trial):
        """Notification for the error of trial.

        This will only be called when the trial is in the RUNNING state."""

        raise NotImplementedError

    def on_trial_result(
        self, tune_controller: "TuneController", trial: Trial, result: Dict
    ) -> str:
        """Called on each intermediate result returned by a trial.

        At this point, the trial scheduler can make a decision by returning
        one of CONTINUE, PAUSE, and STOP. This will only be called when the
        trial is in the RUNNING state."""

        raise NotImplementedError

    def on_trial_complete(
        self, tune_controller: "TuneController", trial: Trial, result: Dict
    ):
        """Notification for the completion of trial.

        This will only be called when the trial is in the RUNNING state and
        either completes naturally or by manual termination."""

        raise NotImplementedError

    def on_trial_remove(self, tune_controller: "TuneController", trial: Trial):
        """Called to remove trial.

        This is called when the trial is in PAUSED or PENDING state. Otherwise,
        call `on_trial_complete`."""

        raise NotImplementedError

    def choose_trial_to_run(self, tune_controller: "TuneController") -> Optional[Trial]:
        """Called to choose a new trial to run.

        This should return one of the trials in tune_controller that is in
        the PENDING or PAUSED state. This function must be idempotent.

        If no trial is ready, return None."""

        raise NotImplementedError

    def debug_string(self) -> str:
        """Returns a human readable message for printing to the console."""

        raise NotImplementedError

    def save(self, checkpoint_path: str):
        """Save trial scheduler to a checkpoint"""
        raise NotImplementedError

    def restore(self, checkpoint_path: str):
        """Restore trial scheduler from checkpoint."""
        raise NotImplementedError


@PublicAPI
class FIFOScheduler(TrialScheduler):
    """Simple scheduler that just runs trials in submission order."""

    def __init__(self):
        super().__init__()

    def on_trial_add(self, tune_controller: "TuneController", trial: Trial):
        pass

    def on_trial_error(self, tune_controller: "TuneController", trial: Trial):
        pass

    def on_trial_result(
        self, tune_controller: "TuneController", trial: Trial, result: Dict
    ) -> str:
        return TrialScheduler.CONTINUE

    def on_trial_complete(
        self, tune_controller: "TuneController", trial: Trial, result: Dict
    ):
        pass

    def on_trial_remove(self, tune_controller: "TuneController", trial: Trial):
        pass

    def choose_trial_to_run(self, tune_controller: "TuneController") -> Optional[Trial]:
        for trial in tune_controller.get_trials():
            if trial.status == Trial.PENDING:
                return trial
        for trial in tune_controller.get_trials():
            if trial.status == Trial.PAUSED:
                return trial
        return None

    def debug_string(self) -> str:
        return "Using FIFO scheduling algorithm."


@DeveloperAPI
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
