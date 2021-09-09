#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import logging
from multiprocessing import process
import time
import uuid
from string import Template
from typing import Dict, List, Any

# Libs
from ray import tune

# Custom
from rest_rpc.training.core.utils import RPCFormatter
from synmanager.train_operations import TrainProducerOperator
from synmanager.completed_operations import CompletedConsumerOperator

##################
# Configurations #
##################

SUPPORTED_METRICS = ['accuracy', 'roc_auc_score', 'pr_auc_score', 'f_score']

rpc_formatter = RPCFormatter()

# Template for generating optimisation run ID
optim_prefix = "optim_run_"
optim_run_template = Template(optim_prefix + "$id")

###########
# Helpers #
###########

class JobCompleted(Exception):
    """ 
    Pseudo-hack to terminate completed consumer loop once an optim cycle 
    has completed.

    Attributes:
        filters (list(str)): Composite IDs of a federated combination
        message (str): Message to be displayed
    """
    def __init__(
        self, 
        filters: List[str],
        outputs: Dict[str, Any],
        message="Optim cycle completed!"
    ):
        self.filters = filters
        self.outputs = outputs
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{' > '.join(self.filters)} -> {self.message}"


def process_results(
    _ids: List[str],
    process: str,
    filters: List[str],
    outputs: Dict[str, Any]
):
    """ Helper function used to curry an archival message from the completed
        queue to check if an optimization cycle has completed, and if so, 
        flags the Tune process for termination
    """
    # Optim Job has gone full cycle! Raise job completion alert
    if process=="validate" and filters == _ids:
        raise JobCompleted(filters=filters, outputs=outputs)

#############
# Functions #
#############

def run_distributed_federated_cycle(
    host: str,
    port: int,
    keys: Dict[str, str],
    grids: List[Dict[str, Any]],
    action: str,
    experiment: Dict[str, Any],
    metric: str,
    auto_align: bool = True,
    dockerised: bool = True, 
    log_msgs: bool = True, 
    verbose: bool = True,
    **params
):
    """ Stores run parameters, train model on specified parameter set, and
        extract validation statistics on validation sets across the federated
        grid.
        Note:
        This function is ALWAYS executed from a local point of reference
        (i.e. TTP not Director). This means that a consumable grid already
        exists and is already pre-allocated.

    Args:
        collab_id (str): Collaboration ID of current collaboration
        project_id (str): Project ID of core project
        expt_id (str): Experiment ID of experimental model architecture
        metric (str): Statistical metric to optimise
        dockerised (bool): Toggles use of dockerised port orchestrations
        log_msgs (bool): Toggles if intermediary operations will be logged
        verbose (bool): Toggles if logging will be started in verbose mode
        **params: Hyperparameter set to train experiment model on
    """  
    # Create an optimisation run under specified experiment for current project
    optim_run_id = optim_run_template.safe_substitute({'id': str(uuid.uuid4())})
    cycle_keys = {**keys, 'run_id': optim_run_id}
    new_optim_run = {
        'key': cycle_keys, 
        'created_at': None, # placeholder
        'relations': {},    # placeholder
        **params
    }

    optim_key, optim_kwargs = list(
        rpc_formatter.enumerate_federated_conbinations(
            action=action,
            experiments=[experiment],
            runs=[new_optim_run],
            auto_align=auto_align,
            dockerised=dockerised,
            log_msgs=log_msgs,
            verbose=verbose
        ).items()
    ).pop()

    producer = TrainProducerOperator(host=host, port=port)
    producer.connect()

    try:
        # Submit parameters of federated combination to job queue
        producer.process(
            process='optimize',   # operations filter for MQ consumer
            keys=optim_key,
            grids=grids,
            parameters={
                'connection': {
                    'host': host,
                    'port': port,
                },
                'hyperparameters': params,
                'info': optim_kwargs
            }
        )

    finally:
        producer.disconnect()


def tune_proc(config: dict, checkpoint_dir: str):
    """ Encapsulating function for Ray.Tune to execute hyperparameter tuning.
        Parameters are dictated by Ray.Tune.

    Args:
        config (dict): 
        checkpoint_dir (str):
    """
    is_cluster = config['is_cluster']

    if not is_cluster:
        raise RuntimeError("Optimization is only active in cluster mode!")

    return run_distributed_federated_cycle(**config)
