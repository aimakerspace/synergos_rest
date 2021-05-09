#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import argparse
import random
import uuid
from string import Template

# Libs
import re
import ray
from ray import tune

# Custom
import synmanager
from rest_rpc import app
from rest_rpc.training.core.server import execute_combination_training
from rest_rpc.training.core.utils import RPCFormatter, Poller
from rest_rpc.evaluation.core.server import execute_combination_inference
from rest_rpc.evaluation.core.utils import MLFlogger
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

is_cluster = app.config['IS_CLUSTER']

db_path = app.config['DB_PATH']
run_records = RunRecords(db_path=db_path)
project_records = ProjectRecords(db_path=db_path)
expt_records = ExperimentRecords(db_path=db_path)
registration_records = RegistrationRecords(db_path=db_path)
model_records = ModelRecords(db_path=db_path)
validation_records = ValidationRecords(db_path=db_path)

rpc_formatter = RPCFormatter()

mlflow_dir = app.config['MLFLOW_DIR']
mlf_logger = MLFlogger()

# Template for generating optimisation run ID
optim_prefix = "optim_run_"
optim_run_template = Template(optim_prefix + "$id")

#############
# Functions #
#############

def run_distributed_federated_cycle(
    producer: TrainProducerOperator,
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
    # Retrieve specific project
    project_keys = {'collab_id': collab_id, 'project_id': project_id}
    retrieved_project = project_records.read(**project_keys)
    project_action = retrieved_project['action']

    # Retrieve specific experiment 
    expt_keys = {**project_keys, 'expt_id': expt_id}
    retrieved_expt = expt_records.read(**expt_keys)
    
    # Create an optimisation run under specified experiment for current project
    optim_run_id = optim_run_template.safe_substitute(
        {'id': optim_prefix + str(uuid.uuid4())}
    )
    cycle_keys ={**expt_keys, 'run_id': optim_run_id}
    run_records.create(**cycle_keys, details=params)
    new_optim_run = run_records.read(**cycle_keys)

    optim_key, optim_kwargs = rpc_formatter.enumerate_federated_conbinations(
        action=project_action,
        experiments=[retrieved_expt],
        runs=[new_optim_run],
        auto_align=auto_align,
        dockerised=dockerised,
        log_msgs=log_msgs,
        verbose=verbose
    ).items().pop()

    # Submit parameters of federated combination to job queue
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
        worker_keys = [participant_id] + optim_key
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


# def start_hp_training(project_id, expt_id, run_id):
#     """ Start hyperparameter training by sending generated hyperparemters 
#         config into the train queue

#     Args:
#         project_id (str)
#         expt_id (str)
#         run_id (str)
#     """
#     # The below train producer logic is extracted from the post function at 
#     # rest_rpc/training/models.py

#     # Populate grid-initialising parameters
#     init_params = {
#         'auto_align': True, 
#         'dockerised': True, 
#         'verbose': True, 
#         'log_msgs': True
#     }

#     # Retrieves expt-run supersets (i.e. before filtering for relevancy)
#     retrieved_project = project_records.read(project_id=project_id)
#     project_action = retrieved_project['action']
#     experiments = retrieved_project['relations']['Experiment']
#     runs = retrieved_project['relations']['Run']

#     # If specific experiment was declared, collapse training space
#     if expt_id:
#         retrieved_expt = expt_records.read(
#             project_id=project_id, 
#             expt_id=expt_id
#         )
#         runs = retrieved_expt['relations']['Run']
#         experiments = [retrieved_expt]

#         # If specific run was declared, further collapse training space
#         if run_id:

#             retrieved_run = run_records.read(
#                 project_id=project_id, 
#                 expt_id=expt_id,
#                 run_id=run_id
#             )
#             runs = [retrieved_run]

#     # Retrieve all participants' metadata
#     registrations = registration_records.read_all(
#         filter={'project_id': project_id}
#     )

#     auto_align = init_params['auto_align']
#     if not auto_align:
#         usable_grids = rpc_formatter.extract_grids(registrations)
#         selected_grid = random.choice(usable_grids)

#         poller = Poller()
#         poller.poll(grid=selected_grid)


#     # Template for starting FL grid and initialising training
#     kwargs = {
#         'action': project_action,
#         'experiments': experiments,
#         'runs': runs,
#         'registrations': registrations
#     }
#     kwargs.update(init_params)

#     # output_payload = None #NOTE: Just added

#     if app.config['IS_CLUSTER_MODE']:
#         train_operator = synmanager.train.TrainProducerOperator(
#             host=app.config["SYN_MQ_HOST"]
#         )
#         result = train_operator.process(project_id, kwargs)

#         #return IDs of runs submitted
#         resp_data = {"run_ids": result}
#         print("resp_data: ", resp_data)  


# def execute_optimization_job(
#     combination_key: List[str],
#     combination_params: Dict[str, Union[str, int, float, list, dict]]
# ) -> List[Document]:


def tune_proc(config: dict, checkpoint_dir: str):
    """ Encapsulating function for Ray.Tune to execute hyperparameter tuning.
        Parameters are dictated by Ray.Tune.

    Args:
        config (dict): 
        checkpoint_dir (str):
    """
    if not is_cluster:
        raise RuntimeError("Optimization is only active in cluster mode!")

    return run_distributed_federated_cycle(**config)


# def send_evaluate_msg(project_id, expt_id, run_id, participant_id=None):
#     """
#         Sending an evaluate message to the evaluate queue given the following args
#         args:
#             project_id: "test_project"
#             expt_id: "test_experiment"
#             run_id: "test_run"
#             participant_id: "test_participant_1"
#     """
#     # Populate grid-initialising parameters
#     # init_params = {'auto_align': True, 'dockerised': True, 'verbose': True, 'log_msgs': True} # request.json
    
#     # Retrieves expt-run supersets (i.e. before filtering for relevancy)
#     retrieved_project = project_records.read(project_id=project_id)
#     print("retrieved_project: ", retrieved_project)
#     project_action = retrieved_project['action']
#     experiments = retrieved_project['relations']['Experiment']
#     runs = retrieved_project['relations']['Run']

#     # If specific experiment was declared, collapse training space
#     if expt_id:

#         retrieved_expt = expt_records.read(
#             project_id=project_id, 
#             expt_id=expt_id
#         )
#         runs = retrieved_expt.pop('relations')['Run']
#         experiments = [retrieved_expt]

#         # If specific run was declared, further collapse training space
#         if run_id:

#             retrieved_run = run_records.read(
#                 project_id=project_id, 
#                 expt_id=expt_id,
#                 run_id=run_id
#             )
#             retrieved_run.pop('relations')
#             runs = [retrieved_run]

#     # Retrieve all participants' metadata
#     registrations = registration_records.read_all(
#         filter={'project_id': project_id}
#     )

#     # Retrieve all relevant participant IDs, collapsing evaluation space if
#     # a specific participant was declared
#     participants = [
#         record['participant']['id'] 
#         for record in registrations
#     ] if not participant_id else [participant_id]

#     # Template for starting FL grid and initialising validation
#     kwargs = {
#         'action': project_action,
#         'experiments': experiments,
#         'runs': runs,
#         'registrations': registrations,
#         'participants': participants,
#         'metas': ['evaluate'],
#         'version': None # defaults to final state of federated grid
#     }

#     # kwargs.update(init_params)

#     if app.config['IS_CLUSTER_MODE']:
#         evaluate_operator = synmanager.evaluate.EvaluateProducerOperator(
#             host=app.config["SYN_MQ_HOST"]
#         )
#         result = evaluate_operator.process(project_id, kwargs)

#         data = {"run_ids": result}


# def start_hp_validations(payload, host):
#     """
#         Custom callback function for sending evaluate message after receiving 
#         the payload from completed queue
#         args:
#             payload:  "TRAINING COMPLETE -  test_project_1/test_experiment_1/optim_run_5c68e185-c28f-4159-8df4-2504ce94f4c7"
#             host: RabbitMQ Server
#     """
    
#     if re.search(r"TRAINING COMPLETE .+/optim_run_.*", payload):
#         message_components = re.findall(r"[\w\-]+", payload)
#         project_id = message_components[3]
#         expt_id = message_components[4]
#         run_id = message_components[5]
#         send_evaluate_msg(project_id, expt_id, run_id)

#     # check if the payload contains training complete before sending to evaluate queue
#     # if message_components[0] == 'TRAINING' and message_components[1] == 'COMPLETE':
#     #     print("STARTING hp validations")
#     #     print(project_id, expt_id, run_id)
#     #     send_evaluate_msg(project_id, expt_id, run_id)
#     else:
#         print("NOT TRAINING. pass..")

# # def read_search_space_path(search_space_path):
# #     '''
# #     Parse search_space.json for project
# #     '''
# #     search_space = json.load(search_space_path)

# #     return search_space

# def str2none(v):
#     '''
#     Converts string None to NoneType for module compatibility
#     in main.py
#     '''
#     if v == "None":
#         return None
#     else:
#         return v
        

# if __name__=='__main__':

#     parser = argparse.ArgumentParser()

#     # receive arguments for synergos mq server host
#     parser.add_argument(
#         '--n_samples',
#         dest='n_samples',
#         help='Synergos HP Tuning',
#         type=int,
#         default=3
#     )

#     # reference where search_space json file is located
#     parser.add_argument(
#         '--search',
#         dest='search_space_path',
#         help='Search space path',
#         type=str
#     )
    
#     args = parser.parse_args()

#     '''
#     search_space = {
#         'algorithm': 'FedProx',
#         'rounds': {"_type": "choice", "_value": [1,2,3,4,5]},
#         'epochs': 1,
#         'lr': 0.001,
#         'weight_decay': 0.0,
#         'lr_decay': 0.1,
#         'mu': 0.1,
#         'l1_lambda': 0.0,
#         'l2_lambda': 0.0,
#         'optimizer': 'SGD',
#         'criterion': 'MSELoss',
#         'lr_scheduler': 'CyclicLR',
#         'delta': 0.0,
#         'patience': 10,
#         'seed': 42,
#         'is_snn': False,
#         'precision_fractional': 5,
#         'base_lr': 0.0005,
#         'max_lr': 0.005,
#     }
#     '''
#     # search_space = read_search_space_path(args.search_space_path)

#     '''
#     kwargs = {
#         "project_id": "test_project_1",
#         "expt_id": "test_experiment_1",
#         "n_samples": args.n_samples,
#         "search_space": search_space,
#     }

#     start_generate_hp(kwargs)
#     '''