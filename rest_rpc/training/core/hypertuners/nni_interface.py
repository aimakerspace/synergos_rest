#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import json
import logging
import os
import shlex
import subprocess
from collections import OrderedDict
from pathlib import Path
from sys import platform
from typing import Dict, List, Tuple, Union

# Libs
import psutil
import yaml
import nnicli

# Custom
from rest_rpc import app
from rest_rpc.training.core.hypertuners import BaseTuner

##################
# Configurations #
##################

SOURCE_FILE = os.path.abspath(__file__)

src_dir = app.config['SRC_DIR']
cores_used = app.config['CORES_USED']
gpu_count = app.config['GPU_COUNT']

##############################################
# Hyperparameter Tuning Interface - NNITuner #
##############################################

class NNITuner(BaseTuner):
    """
    Interfacing class for performing hyperparameter tuning on NNI.
    
    Attributes:
        platform (str): What hyperparameter tuning service to use
        log_dir (str): Directory to export cached log files
    """

    def __init__(self, log_dir: str = None):
        super().__init__(platform="nni", log_dir=log_dir)
        self.__tunable = None

    ############
    # Checkers #
    ############

    def is_running(self) -> bool:
        """ Checks if the execution of current __tunable is still in progress

        Returns:
            State (bool)
        """
        curr_status = self.__tunable.get_experiment_status()['status']
        return curr_status == "RUNNING"

    ###########    
    # Helpers #
    ###########

    @staticmethod
    def _execute_command(command):
        """ Executes commandline operations in Python

        Args:
            command (str): Commandline command to be executed
        Returns:
            Text outputs from executed command
        """
        try:
            completed_process = subprocess.run(
                shlex.split(command),
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            results = completed_process.stdout
            return results

        except subprocess.CalledProcessError as cpe:
            logging.error(f"NNI: Something went wrong during tuning initialisation! {cpe}")
            raise Exception


    def _generate_search_config(
        self, 
        search_space: Dict[str, Dict[str, Union[str, bool, int, float]]]
    ):
        """ Takes in a user-defined search space for tuning a particular 
            process/function and exports it to a JSON file

        Args:
            search_space (dict): Dictionary containing ranges/options to test
                for specified hyperparameter sets compatible with the system
        Returns:
            Path to search configurations (str)
        """
        out_dir = self.generate_output_directory()
        search_config_path = os.path.join(out_dir, "search_space.json")
        os.makedirs(Path(search_config_path).parent, exist_ok=True)

        with open(search_config_path, 'w') as scp:
            json.dump(search_space, scp)
        
        return search_config_path


    def _generate_tuning_config(
        self,
        collab_id: str,
        project_id: str,
        expt_id: str,
        search_config_path: str,
        tuner: str,
        metric: str,
        optimize_mode: str,
        trial_concurrency: int = 1,
        max_exec_duration: str = "1h",
        max_trial_num: int = 10,
        is_remote: bool = True,
        use_annotation: bool = True,
        auto_align: bool = True,
        dockerised: bool = True,
        verbose: bool = True,
        log_msgs: bool = True,
        **kwargs
    ):
        """ Builds configuration YAML file describing current set of experiments

        Args:
            author_name (str):
            expt_name (str):
            search_config_path (str):
            tuner (str):
            optimize_mode (str):
            trial_concurrency (int):
            max_exec_duration (str):
            max_trial_num (int)
            is_remote (bool)
            use_annotation (bool)
        Returns:
            Path to tuning configuration (str)
        """
        out_dir = self.generate_output_directory()

        configurations = OrderedDict()
        configurations['authorName'] = project_id
        configurations['experimentName'] = expt_id
        configurations['logDir'] = out_dir
        configurations['trialConcurrency'] = 1#trial_concurrency
        configurations['maxExecDuration'] = max_exec_duration
        configurations['maxTrialNum'] = max_trial_num
        configurations['trainingServicePlatform'] = "local"#"remote" if is_remote else "local"
        configurations['searchSpacePath'] = search_config_path
        configurations['useAnnotation'] = False#use_annotation
        configurations['tuner'] = {
            'builtinTunerName': tuner,
            'classArgs': {
                'optimize_mode': optimize_mode
            }
        }

        driver_script_path = "rest_rpc.training.core.hypertuners.nni_driver_script"
        configurations['trial'] = {
            'command': "python -m {} -cid {} -pid {} -eid {} -m {} -d -l -v".format(
                driver_script_path,
                collab_id,
                project_id,
                expt_id,
                metric
            ),
            'codeDir': src_dir
            # 'gpuNum': gpu_count,
            # 'cpuNum': cores_used,
            # 'memoryMB': psutil.virtual_memory().available # remaining system ram
        }

        nni_config_path = os.path.join(out_dir, "config.yaml")
        os.makedirs(Path(nni_config_path).parent, exist_ok=True)
        
        with open(nni_config_path, 'w') as ncp:
            yaml.dump(configurations, ncp)

        return nni_config_path

    ##################
    # Core Functions #
    ##################

    def tune(
        self,
        collab_id: str,
        project_id: str,
        expt_id: str,
        search_space: Dict[str, Dict[str, Union[str, bool, int, float]]],
        tuner: str,
        metric: str,
        optimize_mode: str,
        trial_concurrency: int = 1,
        max_exec_duration: str = "1h",
        max_trial_num: int = 10,
        is_remote: bool = True,
        use_annotation: bool = True,
        auto_align: bool = True,
        dockerised: bool = True,
        verbose: bool = True,
        log_msgs: bool = True,
        **kwargs
    ):
        """
        """
        search_config_path = self._generate_search_config(search_space)
        
        nni_config_path = self._generate_tuning_config(
            collab_id=collab_id,
            project_id=project_id,
            expt_id=expt_id,
            search_config_path=search_config_path,
            tuner=tuner,
            metric=metric,
            optimize_mode=optimize_mode,
            trial_concurrency=trial_concurrency,
            max_exec_duration=max_exec_duration,
            max_trial_num=max_trial_num,
            is_remote=is_remote,
            use_annotation=use_annotation
        )

        nnicli.start_nni(config_file=nni_config_path)
        nnicli.set_endpoint('http://0.0.0.0:8080')
        self.__tunable = nnicli

        return self.__tunable