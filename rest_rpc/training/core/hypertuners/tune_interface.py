#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import inspect
import os
from typing import Dict, List, Tuple, Union, Callable, Any

# Libs
import ray
from ray import tune
from ray.tune.ray_trial_executor import RayTrialExecutor
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.basic_variant import BasicVariantGenerator

# Custom
from rest_rpc import app
from rest_rpc.training.core.utils import TuneParser
from rest_rpc.training.core.hypertuners import BaseTuner 
from rest_rpc.training.core.hypertuners.tune_driver_script import tune_proc
from synarchive.connection import RunRecords
from synmanager.train_operations import TrainProducerOperator

##################
# Configurations #
##################

SOURCE_FILE = os.path.abspath(__file__)

is_master = app.config['IS_MASTER']
is_cluster = app.config['IS_CLUSTER']

db_path = app.config['DB_PATH']

cores_used = app.config['CORES_USED']
gpu_count = app.config['GPU_COUNT']

tune_parser = TuneParser()

logging = app.config['NODE_LOGGER'].synlog
logging.debug("training/optimizations.py logged", Description="No Changes")

##################################################
# Hyperparameter Tuning Interface - RayTuneTuner #
##################################################

class RayTuneTuner(BaseTuner):
    """
    Interfacing class for performing hyperparameter tuning on Ray.Tune. Due to
    job parallelization, it is contingent that a queue be enforced, and thus, a
    producer from Synergos Manager is necessary to facilitate this procedure. 
    
    Attributes:

        platform (str): What hyperparameter tuning service to use
        log_dir (str): Directory to export cached log files
    """

    def __init__(self, host: str, port: int, log_dir: str = None):
        super().__init__(platform="tune", log_dir=log_dir)
        self.host = host
        self.port = port

    ############
    # Checkers #
    ############

    def is_running(self) -> bool:
        """ Checks if the execution of current tunable is still in progress

        Returns:
            State (bool)
        """
        has_pending = self.__executor.in_staging_grace_period()
        has_active = len(self.__executor.get_running_trials())
        return has_pending or has_active

    ###########
    # Helpers #
    ###########
    
    @staticmethod
    def _generate_cycle_name(keys: Dict[str, Any]) -> str:
        """ Generates a unique name for the current optimization process

        Args:
            filters (list(str)): Composite IDs of a federated combination
        Returns:
            Cycle name (str)
        """
        collab_id = keys['collab_id']
        project_id =  keys['project_id']
        expt_id = keys['expt_id']
        optim_cycle_name = f"{collab_id}-{project_id}-{expt_id}-optim" 
        return optim_cycle_name


    @staticmethod
    def _retrieve_args(callable: Callable, **kwargs) -> List[str]:
        """ Retrieves all argument keys accepted by a specified callable object
            from a pool of miscellaneous potential arguments

        Args:
            callable (callable): Callable object to be analysed
            kwargs (dict): Any miscellaneous arguments
        Returns:
            Argument keys (list(str))
        """
        input_params = list(inspect.signature(callable).parameters)

        arguments = {}
        for param in input_params:
            param_value = getattr(kwargs, param, None)
            if param_value:
                arguments[param] = param_value

        return arguments


    @staticmethod
    def _count_args(callable: Callable) -> List[str]:
        """ Counts no. of parameters ingestible by a specified callable object.

        Args:
            callable (callable): Callable object to be analysed
        Returns:
            Argument keys (list(str))
        """
        input_params = list(inspect.signature(callable).parameters)
        param_count = len(input_params)
        return param_count


    def _parse_max_duration(self, duration: str = "1h") -> int:
        """ Takes a user specified max duration string and converts it to 
            number of seconds.

        Args:
            duration (str): Max duration string
        Returns:
            Duration in seconds (int) 
        """
        SECS_PER_UNIT = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
        convert_to_seconds = lambda s: int(s[:-1]) * SECS_PER_UNIT[s[-1]]

        try:
            duration_tokens = duration.split(' ')
            total_seconds = sum([convert_to_seconds(d) for d in duration_tokens])
            return total_seconds

        except Exception:
            logging.warn(f"Invalid duration '{duration}' declared! Defaulted to None.")
            return None


    def _calculate_resources(
        self, 
        max_concurrent: int
    ) -> Dict[str, Union[float, int]]:
        """ Distributes no. of system cores to hyperparameter set generation

        Args:
            max_concurrent (int): No. of concurrent Tune jobs to run
        Returns:
            Resource kwargs (dict) 
        """
        ###########################
        # Implementation Footnote #
        ###########################

        # [Causes]
        # Tune is responsible for generating hyperparameter sets. However,
        # Synergos handles all federated training within their own grids.

        # [Problems]
        # Tune auto-detects the amount of resources available for use. This is
        # not an issue if only one Synergos component is deployed onto 1 VM.
        # However, in the event that multiple components are deployed to the
        # same VM, then these Tune jobs would auto-scale & over-allocate itself
        # resources, when in fact it is performing a low-compute task of set
        # generation as compared to other components

        # [Solutions]
        # Only allocate cores available to the system to generate 
        # hyperparameter sets, since all jobs will be handled out of Tune, 
        # in seperate grids, that may or may not be in the same machine
        # consuming the same resources. 
        resources_per_trial={
            'cpu': cores_used/max_concurrent, 
            'gpu': 0
        }

        return resources_per_trial


    def _initialize_trial_scheduler(self, scheduler_str: str, **kwargs):
        """ Parses user inputs and initializes a Tune trial scheduler to manage
            the optimization process

        Args:
            scheduler_str (str): Name of scheduler module as a string
            kwargs: Any parameters as required by aforementioned scheduler
        Returns:
            Scheduler (Tune object) 
        """
        parsed_scheduler = tune_parser.parse_scheduler(scheduler_str=scheduler_str)
        scheduler_args = self._retrieve_args(parsed_scheduler, **kwargs)
        initialized_scheduler = parsed_scheduler(**scheduler_args)
        return initialized_scheduler


    def _initialize_trial_searcher(self, searcher_str: str, **kwargs):
        """ Parses user inputs and initializes a Tune trial searcher to manage
            the optimization process

            Note: 
            Axsearch comflicting dependencies. Dragonfly-opt is not supported

        Args:
            searcher_str (str): Name of searcher module as a string
            kwargs: Any parameters as required by aforementioned searcher
        Returns:
            Searcher (Tune object)
        """
        parsed_searcher = tune_parser.parse_searcher(searcher_str=searcher_str)
        searcher_args = self._retrieve_args(parsed_searcher, **kwargs)
        initialized_searcher = parsed_searcher(**searcher_args)

        ###########################
        # Implementation Footnote #
        ###########################

        # [Causes]
        # BasicVariantGenerator in Tune is fundamental, so it itself is a
        # searcher, and is the primary wrapper around other searchers

        # [Problems]
        # This is asymmetric since it is unable to wrap around itself 
        
        # [Solutions]
        # Detect if intialized searcher is of BasicVariantGenerator, and manage
        # it accordingly
        
        if isinstance(initialized_searcher, BasicVariantGenerator):
            search_algo = initialized_searcher

        else:
            search_algo = ConcurrencyLimiter(
                searcher=initialized_searcher, 
                max_concurrent=1,#kwargs.get('max_concurrent', 1),
                batch=False
            )
        
        return search_algo


    def _initialize_trial_executor(self):
        """ Initializes a trial executor for subsequent use
        """
        self.__executor = RayTrialExecutor(queue_trials=False)
        return self.__executor

    
    def _initialize_tuning_params(
        self,
        optimize_mode: str,
        trial_concurrency: int = 1,
        max_exec_duration: str = "1h",
        max_trial_num: int = 10,
        verbose: bool = False,
        **kwargs
    ) -> Dict[str, Union[float, int]]:
        """ Parses user inputs and generates a set of tuning kwargs to 
            initialize other required parameter objects

        Args:
            optimize_mode (str): Direction to optimize metric (i.e. "max"/"min")
            trial_concurrency (int): No. of trials to run concurrently. This is
                in context of Tune, and is independent to the no. of jobs that
                can be run concurrently across Synergos grids. Eg. Tune is
                supposed to create 10 trials, at 5 concurrently, but 
                collaborators have only deployed 2 usable grids. This way, Tune
                will still generate 5 trials, but will have to wait until all
                5 trials have been completed across 2 Synergos grids first
                before proceeding on to the next batch of 5 trials
            max_exec_duration (str): Duration string capping each trial's runtime
            max_trial_num (int): Max number of trials to run before giving up
            verbose (bool): Toggles verbosity of outputs
            kwargs: Miscellaneous params that may or may not be used
        Returns:
            Tuning parameters (dict)
        """
        parsed_duration = self._parse_max_duration(max_exec_duration)
        configured_executor = self._initialize_trial_executor()
        local_dir = self.generate_output_directory()
        return {
            'mode': optimize_mode,
            'num_samples': max_trial_num,
            'time_budget_s': parsed_duration,
            'trial_executor': configured_executor,
            'local_dir': local_dir,
            'checkpoint_at_end': False,
            'verbose': 3 if verbose else 1,
            'log_to_file': True
        }


    def _initialize_search_space(self, search_space: dict) -> dict:
        """ Mapping custom search space config into Tune config

        Args:
            search_space (dict): Parameter space to search upon
        Returns
            Tune search configurations (dict)
        """
        configured_search_space = {}
        for hyperparameter_key in search_space.keys():

            hyperparameter_type = search_space[hyperparameter_key]['_type']
            hyperparameter_value = search_space[hyperparameter_key]['_value']
            
            try:
                parsed_type = tune_parser.parse_type(hyperparameter_type)
                param_count = self._count_args(parsed_type)
                tune_config_value = (
                    parsed_type(*hyperparameter_value) 
                    if param_count > 1 
                    else parsed_type(hyperparameter_value)
                )
                configured_search_space[hyperparameter_key] = tune_config_value

            except:
                raise RuntimeError(f"Specified hyperparmeter type '{hyperparameter_type}' is unsupported!")

        return configured_search_space

    ##################
    # Core Functions #
    ##################

    def tune(
        self,
        keys: Dict[str, str],
        grids: List[Dict[str, Any]],
        action: str,
        experiment: Dict[str, Any],
        search_space: Dict[str, Dict[str, Union[str, bool, int, float]]],
        metric: str,
        optimize_mode: str,
        scheduler: str = "AsyncHyperBandScheduler",
        searcher: str = "BasicVariantGenerator",
        trial_concurrency: int = 1,
        max_exec_duration: str = "1h",
        max_trial_num: int = 10,
        auto_align: bool = True,
        dockerised: bool = True,
        verbose: bool = True,
        log_msgs: bool = True,
        **kwargs
    ):
        """ Triggers parallelized hyperparameter optimzation

        Args:
            keys (dict): Unique IDs that form a composite key identifying the
                current federated cycle
            grids (list(dict)): All availble registered Synergos grids
            action (str): ML operation to perform
            experiment (dict): Experiment record documenting model architecture
            search_space (dict): Parameter space to search upon
            metric (str): Metric to optimize on
            optimize_mode (str): Direction to optimize metric (i.e. "max"/"min")
            scheduler (str): Name of scheduler module as a string
            searcher (str): Name of searcher module as a string
            trial_concurrency (int): No. of Tune trials to generate concurrently
            max_exec_duration (str): Duration string capping each trial's runtime
            max_trial_num (int): Max number of trials to run before giving up
            auto_align (bool): Toggles if model should be auto-aligned
            dockerized (bool): Toggles if deployed system is dockerized
            verbose (bool): Toggles verbosity of outputs
            log_msgs (bool): Toggles logging
            kwargs: Miscellaneous parameters
        Returns:
            Results
        """
        ###########################
        # Implementation Footnote #
        ###########################

        # [Cause]
        # In SynCluster mode, all processes are inducted as jobs. All jobs are sent
        # to Synergos MQ to be linearized for parallel distributed computing.

        # [Problems]
        # Ray logs need to be aligned across distributed setting

        # [Solution]
        # Start director as a ray head node, with all other TTPs as child nodes 
        # connecting to it. Tuning parameters will be reported directly to the head
        # node, bypassing the queue

        ray.init(num_cpus=cores_used, num_gpus=0)
        assert ray.is_initialized() == True

        try:
            optim_cycle_name = self._generate_cycle_name(keys)

            configured_search_space = self._initialize_search_space(search_space)

            config = {
                'is_cluster': is_cluster,
                'host': self.host,
                'port': self.port,
                'keys': keys,
                'grids': grids,
                'action': action,
                'experiment': experiment,
                'metric': metric,
                'auto_align': auto_align,
                'dockerised': dockerised,
                'verbose': verbose,
                'log_msgs': log_msgs,
                **configured_search_space
            }

            configured_resources = self._calculate_resources(max_trial_num)

            tuning_params = self._initialize_tuning_params(
                optimize_mode=optimize_mode,
                trial_concurrency=trial_concurrency,
                max_exec_duration=max_exec_duration,
                max_trial_num=max_trial_num,
                verbose=verbose,
                **kwargs
            )

            configured_scheduler = self._initialize_trial_scheduler(
                scheduler_str=scheduler, 
                **{**kwargs, **tuning_params}
            )

            search_algorithm = self._initialize_trial_searcher(
                searcher_str=searcher,
                **{**kwargs, **tuning_params}
            )

            results = tune.run(
                tune_proc, 
                name=optim_cycle_name,
                config=config, 
                resources_per_trial=configured_resources,
                scheduler=configured_scheduler,
                search_alg=search_algorithm,
                **tuning_params
            )

        finally:
            # Stop Ray instance
            ray.shutdown()
            assert ray.is_initialized() == False

        return results
