#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import uuid
from string import Template

# Libs
from ray import tune

# Custom
from rest_rpc.training.core.utils import RPCFormatter
from synarchive.connection import (
    ProjectRecords,
    ExperimentRecords,
    RunRecords,
    RegistrationRecords
)
from synarchive.training import ModelRecords
from synarchive.evaluation import ValidationRecords
from synmanager.train_operations import TrainProducerOperator
# from synmanager.completed import CompletedConsumerOperator

##################
# Configurations #
##################

SUPPORTED_METRICS = ['accuracy', 'roc_auc_score', 'pr_auc_score', 'f_score']

rpc_formatter = RPCFormatter()

# Template for generating optimisation run ID
optim_prefix = "optim_run_"
optim_run_template = Template(optim_prefix + "$id")

#############
# Functions #
#############

def run_distributed_federated_cycle(
    host: str,
    port: int,
    db_path: str,
    collab_id: str,
    project_id: str,
    expt_id: str,
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
    run_records = RunRecords(db_path=db_path)
    project_records = ProjectRecords(db_path=db_path)
    expt_records = ExperimentRecords(db_path=db_path)
    registration_records = RegistrationRecords(db_path=db_path)
    model_records = ModelRecords(db_path=db_path)
    validation_records = ValidationRecords(db_path=db_path)

    # Retrieve specific project
    project_keys = {'collab_id': collab_id, 'project_id': project_id}
    retrieved_project = project_records.read(**project_keys)
    project_action = retrieved_project['action']

    # Retrieve specific experiment 
    expt_keys = {**project_keys, 'expt_id': expt_id}
    retrieved_expt = expt_records.read(**expt_keys)
    
    # Create an optimisation run under specified experiment for current project
    optim_run_id = optim_run_template.safe_substitute({'id': str(uuid.uuid4())})
    cycle_keys ={**expt_keys, 'run_id': optim_run_id}
    run_records.create(**cycle_keys, details=params)
    new_optim_run = run_records.read(**cycle_keys)

    optim_key, optim_kwargs = list(
        rpc_formatter.enumerate_federated_conbinations(
            action=project_action,
            experiments=[retrieved_expt],
            runs=[new_optim_run],
            auto_align=auto_align,
            dockerised=dockerised,
            log_msgs=log_msgs,
            verbose=verbose
        ).items()
    ).pop()

    # Submit parameters of federated combination to job queue
    producer = TrainProducerOperator(host=host, port=port)
    producer.connect()
    producer.process(
        process='optimize',   # operations filter for MQ consumer
        combination_key=optim_key,
        combination_params=optim_kwargs
    )

    ###########################
    # Implementation Footnote #
    ###########################

    # [Cause]
    # PySyft Grids cannot host more than 1 federated cycle at at time. Hence, 
    # to allow for active grid control, all optimization cycles are to be 
    # channelled into a message queue.

    # [Problems]
    # In order for Ray.Tune to invoke its hyperparameter/scheduling 
    # capabilities, it can only detect statistics from within its own sessions,
    # which refers only to the direct function instances called from executing
    # `tune.run()`. The queue cannot be bypassed by using Ray Nodes, since
    # there may be cases where optimization runs are ran alongside other 
    # training/evaluation jobs, causing the problem of conflicting grids.

    # [Solution]
    # Since jobs will perform archival processes as well, Director is to wait
    # until statistics for a job exists in the archive before continuing.

    registrations = registration_records.read_all(filter=project_keys)
    participants = [record['participant']['id'] for record in registrations]

    grouped_statistics = {}
    while participants:

        participant_id = participants.pop(0)
        worker_keys = [participant_id] + list(optim_key)
        inference_stats = validation_records.read(*worker_keys)

        # Archival for current participant is completed
        if inference_stats:

            # Culminate into collection of metrics
            for metric_opt in SUPPORTED_METRICS:
                metric_collection = grouped_statistics.get(metric_opt, [])
                curr_metrics = inference_stats['evaluate']['statistics'][metric_opt]
                metric_collection.append(curr_metrics)
                grouped_statistics[metric_opt] = metric_collection

        # Archival for current participant is still pending --> Postpone
        else:
            participants.append(participant_id)

    # Calculate average of all statistics as benchmarks for model performance
    process_nans = lambda x: [max(stat, 0) for stat in x]
    calculate_avg_stats = lambda x: (sum(x)/len(x)) if x else 0
    avg_statistics = {
        metric: calculate_avg_stats(process_nans([
            calculate_avg_stats(process_nans(p_metrics)) 
            for p_metrics in metric_collection
        ]))
        for metric, metric_collection in grouped_statistics.items()
    }

    tune.report(**avg_statistics)


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


