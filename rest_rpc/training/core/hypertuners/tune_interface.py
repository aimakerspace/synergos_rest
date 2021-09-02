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
        """
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


    def _calculate_resources(self):
        """
        """
        # If TTP node is master node --> Synergos Basic --> No parallelisation!
        if is_master:

            ###########################
            # Implementation Footnote #
            ###########################

            # Pseudo-parallelisation lock: Allocate all resources to a single 
            # trial 
            resources_per_trial={
                'cpu': cores_used,
                # 'gpu': gpu_count
            }

            return resources_per_trial


    def _initialize_trial_scheduler(self, scheduler_str: str, **kwargs):
        """
        """
        parsed_scheduler = tune_parser.parse_scheduler(scheduler_str=scheduler_str)
        scheduler_args = self._retrieve_args(parsed_scheduler, **kwargs)
        initialized_scheduler = parsed_scheduler(**scheduler_args)
        return initialized_scheduler


    def _initialize_trial_searcher(self, searcher_str: str, **kwargs):
        """ Axsearch comflicting dependencies. Dragonfly-opt is not supported
        """
        parsed_searcher = tune_parser.parse_searcher(searcher_str=searcher_str)
        searcher_args = self._retrieve_args(parsed_searcher, **kwargs)
        initialized_searcher = parsed_searcher(**searcher_args)

        ###########################
        # Implementation Footnote #
        ###########################

        if isinstance(initialized_searcher, BasicVariantGenerator):
            search_algo = initialized_searcher

        else:
            search_algo = ConcurrencyLimiter(
                searcher=initialized_searcher, 
                max_concurrent=2,
                batch=False
            )
        
        return search_algo


    def _initialize_trial_executor(self):
        """
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
    ):
        """
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


    def _initialize_search_space(self, search_space: dict):
        """ Mapping custom search space config into tune config

        """
        logging.warning(f"--->>> search space: {search_space}")
        configured_search_space = {}
        for hyperparameter_key in search_space.keys():


            hyperparameter_type = search_space[hyperparameter_key]['_type']
            hyperparameter_value = search_space[hyperparameter_key]['_value']
            
            # try:
            parsed_type = tune_parser.parse_type(hyperparameter_type)
            param_count = self._count_args(parsed_type)
            tune_config_value = (
                parsed_type(*hyperparameter_value) 
                if param_count > 1 
                else parsed_type(hyperparameter_value)
            )
            configured_search_space[hyperparameter_key] = tune_config_value

            # except:
            #     raise RuntimeError(f"Specified hyperparmeter type '{hyperparameter_type}' is unsupported!")

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
        """
        """
        ###########################
        # Implementation Footnote #
        ###########################

        # [Cause]
        # In SynCluster mode, all processes are inducted as jobs. All jobs are sent
        # to Synergos MQ to be linearized for parallel distributed computing.

        # [Problems]
        # 

        # [Solution]
        # Start director as a ray head node, with all other TTPs as child nodes 
        # connecting to it. Tuning parameters will be reported directly to the head
        # node, bypassing the queue

        # ray.init()
        # assert ray.is_initialized() == True

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

            configured_resources = self._calculate_resources()

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
            # ray.shutdown()
            # assert ray.is_initialized() == False
            pass

        return results
