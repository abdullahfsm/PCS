import logging
from typing import Dict, Optional, Union
import sys

import numpy as np
import pickle

from ray.tune import trial_runner
from ray.tune.result import DEFAULT_METRIC
from ray.tune.schedulers.trial_scheduler import FIFOScheduler, TrialScheduler
from ray.tune.trial import Trial

logger = logging.getLogger(__name__)



class SyncSuccessiveHalving(FIFOScheduler):
    """Implements the sync Successive Halving.

    This should provide similar theoretical performance as HyperBand but
    avoid straggler issues that HyperBand faces. One implementation detail
    is when using multiple brackets, trial allocation to bracket is done
    randomly with over a softmax probability.

    See https://arxiv.org/abs/1810.05934

    this is a tester

    Args:
        time_attr (str): A training result attr to use for comparing time.
            Note that you can pass in something non-temporal such as
            `training_iteration` as a measure of progress, the only requirement
            is that the attribute should increase monotonically.
        metric (str): The training result objective value attribute. Stopping
            procedures will use this attribute. If None but a mode was passed,
            the `ray.tune.result.DEFAULT_METRIC` will be used per default.
        mode (str): One of {min, max}. Determines whether objective is
            minimizing or maximizing the metric attribute.
        max_t (float): max time units per trial. Trials will be stopped after
            max_t time units (determined by time_attr) have passed.
        min_budget (float): Only stop trials at least this old in time.
            The units are the same as the attribute named by `time_attr`.
        reduction_factor (float): Used to set halving rate and amount. This
            is simply a unit-less scalar.
    """

    # def __init__(self,
    #             time_attr: str = "training_iteration",
    #             metric: Optional[str] = None,
    #             mode: Optional[str] = "min",
    #             budget: float = 100.0,
    #             num_samples: int = 10,
    #             reduction_factor: float = 3):


    def __init__(self,
                time_attr: str = "training_iteration",
                metric: Optional[str] = None,
                mode: Optional[str] = "min",
                budget: float = 100.0,
                num_samples: int = 10,
                reduction_factor: float = 3,
                temporal_increase: bool = True):


        assert budget > 0, "Budget should be +tive!"
        assert num_samples > 0, "Number of samples should be positive"
        assert reduction_factor > 1, "Reduction Factor not valid!"
        
        if mode:
            assert mode in ["min", "max"], "`mode` must be 'min' or 'max'!"


        FIFOScheduler.__init__(self)
        self._budget = budget
        self._num_samples = num_samples
        self._reduction_factor = reduction_factor

        self._trial_pause_list = set()



        self._total_stages = int(np.floor(np.log(self._num_samples)/np.log(self._reduction_factor))) + 1
        

        self._min_t = self._budget

        # self._min_t = max(int(self._budget/(self._total_stages * self._num_samples)),1.0)

        self._cur_stage = 0
        self._stage_candidates = []
        self._stage_budget = []
        n0 = self._num_samples
        t0 = self._min_t
        time_budget = t0
        for _ in range(self._total_stages):
            self._stage_candidates.append(n0)
            self._stage_budget.append(time_budget)


            if n0 == 1:
                break

            n0 = n0 // self._reduction_factor
            if temporal_increase:
                t0 = t0 * self._reduction_factor

            time_budget += t0


        if self._stage_candidates[-1] > 1:
            self._stage_candidates.append(1)
            self._stage_budget.append(time_budget)
            self._total_stages+=1




        


        ########## DEBUG ##############
        print("Initialising SHA scheduler")
        print("budget: %d\tnum_samples:%d\treduction_factor:%d" % (self._budget, self._num_samples, self._reduction_factor))
        print("total_stages: %d\tmin_t: %d" % (self._total_stages, self._min_t))
        for stage in range(self._total_stages):
            print("stage %d\tnum_samples:%d\ttime:%d" % (stage, self._stage_candidates[stage], self._stage_budget[stage]))


        assert self._stage_candidates[-1] == 1, "Last stage has more than 1 candidate"
        assert len(self._stage_candidates) == self._total_stages


        # maintain counters for trials
        self._trial_info = {}  # Stores trial_id -> trial, score, iteration, status
        self._samples_added = 0
        self._num_paused = 0  # for
        self._num_stopped = 0
        self._metric = metric
        self._mode = mode
        self._metric_op = None
        if self._mode == "max":
            self._metric_op = 1.
        elif self._mode == "min":
            self._metric_op = -1.
        self._time_attr = time_attr

    def __cur_stage_candidates(self):
        return self._stage_candidates[self._cur_stage]
    def __cur_stage_budget(self):
        return self._stage_budget[self._cur_stage]


    def __stage_candidates(self, stage):
        if stage < self._total_stages:
            return self._stage_candidates[stage]
        return None


    def __stage_budget(self, stage):
        if stage < self._total_stages:
            return self._stage_budget[stage]
        return None


    def set_search_properties(self, metric: Optional[str],
                              mode: Optional[str]) -> bool:
        if self._metric and metric:
            return False
        if self._mode and mode:
            return False

        if metric:
            self._metric = metric
        if mode:
            self._mode = mode

        if self._mode == "max":
            self._metric_op = 1.
        elif self._mode == "min":
            self._metric_op = -1.

        if self._metric is None and self._mode:
            # If only a mode was passed, use anonymous metric
            self._metric = DEFAULT_METRIC

        return True


    def add_trial_to_pause_list(self, trial: Trial):
        self._trial_pause_list.add(trial)


    def remove_trial_from_pause_list(self, trial: Trial):
        self._trial_pause_list.remove(trial)



    def on_trial_add(self, trial_runner: "trial_runner.TrialRunner",
                     trial: Trial):

        self._samples_added += 1

        if self._samples_added > self._num_samples:
            print("Issue: more than promised trials are trying to run")



        if not self._metric or not self._metric_op:
            raise ValueError(
                "{} has been instantiated without a valid `metric` ({}) or "
                "`mode` ({}) parameter. Either pass these parameters when "
                "instantiating the scheduler, or pass them as parameters "
                "to `tune.run()`".format(self.__class__.__name__, self._metric,
                                         self._mode))

        # trial, score, iteration, status

        print("DEBUG: added %s to self._trial_info" % trial.trial_id)

        self._trial_info[trial.trial_id] = {"trial": trial, "score": None, "iteration": None, "status": "ACTIVE"}
        

    def on_trial_result(self, trial_runner: "trial_runner.TrialRunner",
                        trial: Trial, result: Dict) -> str:


        # print("num_paused: %d" % self._num_paused)

        # check if the budget has elapsed


        # if trial in self._trial_pause_list:
        #     return TrialScheduler.PAUSE_YEILD


        if result[self._time_attr] < self.__cur_stage_budget():
            action = TrialScheduler.CONTINUE
        else:


            action = TrialScheduler.PAUSE
            # self._trial_info[trial.trial_id][1] = self._metric_op * result[self._metric]

            if result[self._metric] == None:
                print("GOT NONE")
            else:
                pass

            if (self._trial_info[trial.trial_id]["status"] != "PAUSE"):
                self._trial_info[trial.trial_id]["score"] = result[self._metric]
                self._trial_info[trial.trial_id]["iteration"] = result[self._time_attr]
                self._trial_info[trial.trial_id]["status"] = "PAUSE"
                # print("Trial_id: %s Loss: %f iteration: %d status: %s" % (trial.trial_id, result[self._metric], result[self._time_attr], self._trial_info[trial.trial_id]["status"]))
                self._num_paused += 1
                # print("%d/%d paused" % (self._num_paused, self.__cur_stage_candidates()))


        # time to cut
        if self._num_paused == self.__cur_stage_candidates() and self._cur_stage < self._total_stages - 1:

            '''
            print("*****************************************")
            print("*****************************************")
            print("*****************************************")
            print("*****************************************")
            for trial_id in self._trial_info:
                print("Trial_id: %s iteration: %s" % (trial_id, str(self._trial_info[trial_id]["iteration"])))

            print("*****************************************")
            print("*****************************************")
            print("*****************************************")
            print("*****************************************")
            '''





            assert(len(self._trial_info) > 1), len(self._trial_info)
            # assert(len(self._trial_info) == self.__cur_stage_candidates()), "%d, %d" % (len(self._trial_info), self.__cur_stage_candidates())

            '''
            print("**********************$$$$$$$$$$************************")

            for t in self._trial_info:
                print(self._trial_info[t][1])

            print("**********************$$$$$$$$$$$$$************************")
            '''

            sorted_trials = sorted(
                self._trial_info,
                key=lambda t: self._metric_op * self._trial_info[t]["score"])

            good, bad = sorted_trials[-self.__stage_candidates(self._cur_stage+1):], sorted_trials[:-self.__stage_candidates(self._cur_stage+1)]

    


            trial_ids_to_stop = bad[:]
            trial_ids_to_continue = good[:]


            for trial_id in trial_ids_to_continue:

                self._trial_info[trial_id]["status"] = "ACTIVE"


                # no need to unpause THIS trial
                if trial.trial_id == trial_id:
                    action = TrialScheduler.CONTINUE
                    continue
                

                # only unpause trials that are not in the preempted_trials
                # these will be unpaused when resources are available
                if not (self._trial_info[trial_id]["trial"] in trial_runner._preempted_trials):
                    trial_runner.trial_executor.unpause_trial(self._trial_info[trial_id]["trial"])
                    logger.info("DEBUG (sync_successive_halving.py): unpausing trial: %s. trial.checkpoint_manager._newest_memory_checkpoint.result[training_iteration] : %s" % 
                        (trial_id, self._trial_info[trial_id]["trial"].checkpoint_manager._newest_memory_checkpoint.result["training_iteration"]))

                # self._trial_info[trial_id]["trial"].status = Trial.PENDING
                # trial_runner.trial_executor.unpause_trial(self._trial_info[trial_id][0])

            for trial_id in trial_ids_to_stop:
                
                if trial.trial_id == trial_id:
                    
                    # print("self._cur_stage: %d" % self._cur_stage)
                    # print("I (%s) am told to stop" % trial.trial_id)

                    # trial_runner.trial_executor.pause_trial(trial)
                    action = TrialScheduler.STOP
                    continue
                
                trial_runner.stop_trial(self._trial_info[trial_id]["trial"])



            # assert len(self._trial_info) == self.__cur_stage_candidates() + 1, "%d vs %d" % (len(self._trial_info),self.__cur_stage_candidates())

            self._num_paused = 0
            self._cur_stage += 1


        elif self._num_paused == self.__cur_stage_candidates() and self._cur_stage == self._total_stages - 1:
            # trial_runner.trial_executor.pause_trial(trial)
            action = TrialScheduler.STOP


        # print("trial_id: %s is told to %s" % (trial.trial_id, action))


        return action

    def on_trial_complete(self, trial_runner: "trial_runner.TrialRunner",
                          trial: Trial, result: Dict):
        if self._time_attr not in result or self._metric not in result:
            return
        # bracket = self._trial_info[trial.trial_id]
        # bracket.on_result(trial, result[self._time_attr],
        #                   self._metric_op * result[self._metric])

        # print("*********deleting %s as it has completed" % trial.trial_id)
        del self._trial_info[trial.trial_id]

    def on_trial_remove(self, trial_runner: "trial_runner.TrialRunner",
                        trial: Trial):

        # print("removing trial_id: %s" % trial.trial_id)

        '''
        for trial_id in self._trial_info:
            print(self._trial_info[trial_id])
        '''

        del self._trial_info[trial.trial_id]


    # only run pending trials
    def choose_trial_to_run(
            self, trial_runner: "trial_runner.TrialRunner") -> Optional[Trial]:
        for trial in trial_runner.get_trials():

            '''
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("trial_id: %s is pending" % trial.trial_id)
            print("has resources for trial: %s" % str(trial_runner.has_resources_for_trial(trial)))
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
            '''

            if (trial.status == Trial.PENDING and trial_runner.has_resources_for_trial(trial)):


                # print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
                # print("trial_id: %s is pending" % trial.trial_id)
                # print("has resources for trial: %s" % str(trial_runner.has_resources_for_trial(trial)))
                # print("max gpu cap: %d" % int(trial_runner.trial_executor._max_GPU))
                # print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
                




                return trial
        return None

    '''
    def debug_string(self) -> str:
        out = "Using AsyncHyperBand: num_stopped={}".format(self._num_stopped)
        out += "\n" + "\n".join([b.debug_str() for b in self._brackets])
        return out
    '''
    def save(self, checkpoint_path: str):
        save_object = self.__dict__
        with open(checkpoint_path, "wb") as outputFile:
            pickle.dump(save_object, outputFile)

    def restore(self, checkpoint_path: str):
        with open(checkpoint_path, "rb") as inputFile:
            save_object = pickle.load(inputFile)
        self.__dict__.update(save_object)

'''
class _Bracket():
    """Bookkeeping system to track the cutoffs.

    Rungs are created in reversed order so that we can more easily find
    the correct rung corresponding to the current iteration of the result.

    Example:
        >>> b = _Bracket(1, 10, 2, 0)
        >>> b.on_result(trial1, 1, 2)  # CONTINUE
        >>> b.on_result(trial2, 1, 4)  # CONTINUE
        >>> b.cutoff(b._rungs[-1][1]) == 3.0  # rungs are reversed
        >>> b.on_result(trial3, 1, 1)  # STOP
        >>> b.cutoff(b._rungs[3][1]) == 2.0
    """

    def __init__(self, min_t: int, max_t: int, reduction_factor: float,
                 s: int):
        self.rf = reduction_factor
        MAX_RUNGS = int(np.log(max_t / min_t) / np.log(self.rf) - s + 1)
        self._rungs = [(min_t * self.rf**(k + s), {})
                       for k in reversed(range(MAX_RUNGS))]

    def cutoff(self, recorded) -> Union[None, int, float, complex, np.ndarray]:
        if not recorded:
            return None
        return np.nanpercentile(
            list(recorded.values()), (1 - 1 / self.rf) * 100)

    def on_result(self, trial: Trial, cur_iter: int,
                  cur_rew: Optional[float]) -> str:
        action = TrialScheduler.CONTINUE
        for milestone, recorded in self._rungs:
            if cur_iter < milestone or trial.trial_id in recorded:
                continue
            else:
                cutoff = self.cutoff(recorded)
                if cutoff is not None and cur_rew < cutoff:
                    action = TrialScheduler.STOP
                if cur_rew is None:
                    logger.warning("Reward attribute is None! Consider"
                                   " reporting using a different field.")
                else:
                    recorded[trial.trial_id] = cur_rew
                break
        return action

    def debug_str(self) -> str:
        # TODO: fix up the output for this
        iters = " | ".join([
            "Iter {:.3f}: {}".format(milestone, self.cutoff(recorded))
            for milestone, recorded in self._rungs
        ])
        return "Bracket: " + iters
'''
