#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import os
from string import Template
from typing import Dict, List, Union, Any

# Libs
from flask import request
from flask_restx import Namespace, Resource, fields
from tinydb.database import Document

# Custom
from rest_rpc import app
from rest_rpc.connection.core.utils import TopicalPayload
from rest_rpc.training.core.hypertuners import (
    NNITuner, 
    RayTuneTuner, 
    optim_prefix
)
from rest_rpc.training.core.utils import RPCFormatter
from rest_rpc.evaluation.core.utils import MLFlogger
from synarchive.connection import (
    CollaborationRecords,
    ProjectRecords,
    ExperimentRecords,
    RunRecords,
    RegistrationRecords
)
from synarchive.training import ModelRecords
from synarchive.evaluation import ValidationRecords, MLFRecords
from synmanager.preprocess_operations import PreprocessProducerOperator
from synmanager.train_operations import TrainProducerOperator
from synmanager.evaluate_operations import EvaluateProducerOperator

##################
# Configurations #
##################

SOURCE_FILE = os.path.abspath(__file__)

SUBJECT = "Optimization"

HYPERTUNER_BACKENDS = {'nni': NNITuner, 'tune': RayTuneTuner}

ns_api = Namespace(
    "optimizations", 
    description='API to faciliate hyperparameter tuning in a federated grid.'
)

grid_idx = app.config['GRID']

is_cluster = app.config['IS_CLUSTER']

out_dir = app.config['OUT_DIR']

db_path = app.config['DB_PATH']
collab_records = CollaborationRecords(db_path=db_path)
project_records = ProjectRecords(db_path=db_path)
expt_records = ExperimentRecords(db_path=db_path)
run_records = RunRecords(db_path=db_path)
mlf_records = MLFRecords(db_path=db_path)
registration_records = RegistrationRecords(db_path=db_path)
model_records = ModelRecords(db_path=db_path)
validation_records = ValidationRecords(db_path=db_path)

rpc_formatter = RPCFormatter()

mlf_logger = MLFlogger()

# Template for generating optimisation run ID
optim_prefix = "optim_run_"
optim_run_template = Template(optim_prefix + "$id")

logging = app.config['NODE_LOGGER'].synlog
logging.debug("training/optimizations.py logged", Description="No Changes")

###########################################################
# Models - Used for marshalling (i.e. moulding responses) #
###########################################################

# Marshalling inputs
tuning_model = ns_api.model(
    name="tuning_input",
    model={
        'search_space': fields.Raw(required=True),
        'tuner': fields.String(),
        'optimize_mode': fields.String(),
        'trial_concurrency': fields.Integer(default=1),
        'max_exec_duration': fields.String(default="1h"),
        'max_trial_num': fields.Integer(default=10),
        'is_remote': fields.Boolean(default=True),
        'use_annotation': fields.Boolean(default=True),
        'dockerised': fields.Boolean(default=False, required=True),
        'verbose': fields.Boolean(default=False),
        'log_msgs': fields.Boolean(default=False)
    }
)

# Marshalling Outputs
# - same `val_output_model` retrieved from Validations resource
# Marshalling inputs 
input_model = ns_api.model(
    name="validation_input",
    model={
        'dockerised': fields.Boolean(default=False, required=True),
        'verbose': fields.Boolean(default=False),
        'log_msgs': fields.Boolean(default=False)
    }
)

# Marshalling Outputs
stats_model = ns_api.model(
    name="statistics",
    model={
        'R2': fields.Float(),
        'MSE': fields.Float(),
        'MAE': fields.Float(),
        'accuracy': fields.List(fields.Float()),
        'roc_auc_score': fields.List(fields.Float()),
        'pr_auc_score': fields.List(fields.Float()),
        'f_score': fields.List(fields.Float()),
        'TPRs': fields.List(fields.Float()),
        'TNRs': fields.List(fields.Float()),
        'PPVs': fields.List(fields.Float()),
        'NPVs': fields.List(fields.Float()),
        'FPRs': fields.List(fields.Float()),
        'FNRs': fields.List(fields.Float()),
        'FDRs': fields.List(fields.Float()),
        'TPs': fields.List(fields.Integer()),
        'TNs': fields.List(fields.Integer()),
        'FPs': fields.List(fields.Integer()),
        'FNs': fields.List(fields.Integer())
    }
)

meta_stats_model = ns_api.model(
    name="meta_statistics",
    model={
        'statistics': fields.Nested(stats_model, skip_none=True),
        'res_path': fields.String(skip_none=True)
    }
)

val_inferences_model = ns_api.model(
    name="validation_inferences",
    model={
        'evaluate': fields.Nested(meta_stats_model, required=True)
    }
)

val_output_model = ns_api.inherit(
    "validation_output",
    val_inferences_model,
    {
        'doc_id': fields.String(),
        'kind': fields.String(),
        'key': fields.Nested(
            ns_api.model(
                name='key',
                model={
                    'participant_id': fields.String(),
                    'collab_id': fields.String(),
                    'project_id': fields.String(),
                    'expt_id': fields.String(),
                    'run_id': fields.String()
                }
            ),
            required=True
        )
    }
)

payload_formatter = TopicalPayload(SUBJECT, ns_api, val_output_model)

########
# Jobs #
########

def execute_optimization_job(
    keys: List[str],
    grids: List[Dict[str, Any]],
    parameters: Dict[str, Union[str, int, float, list, dict]]
) -> List[Document]:
    """ Encapsulated job function to be compatible for queue integrations.
        Executes model training & inference for a specified federated cycle, 
        to optimize.

    Args:
        keys (list(str)): IDs related to federated job 
        grid (list(dict))): Registry of participants' node information
        parameters (dict): Initializing parameters for a federated job
    Returns:
        Optimized validation statistics (list(Document))    
    """   
    mq_host = parameters['connection']['host']
    mq_port = parameters['connection']['port'] 
    job_info = parameters['info']

    # preprocess_producer = PreprocessProducerOperator(host=mq_host, port=mq_port)
    train_producer = TrainProducerOperator(host=mq_host, port=mq_port)
    evaluate_producer = EvaluateProducerOperator(host=mq_host, port=mq_port)

    # preprocess_producer.connect()
    train_producer.connect()
    evaluate_producer.connect()

    try:
        # Step 1 -> Phase 2B: Train on experiment-run combination
        train_producer.process(
            process='train',
            keys=keys,
            grids=grids,
            parameters=job_info
        )

        # Step 2 -> Phase 3A: Calculate validation statistics for target combination
        selected_grid = grids[grid_idx]
        participants = [registry['keys']['participant_id'] for registry in selected_grid]
        evaluate_producer.process(
            process='validate',
            keys=keys,
            grids=grids,
            parameters={**job_info, 'participants': participants}
        )

    finally:
        # preprocess_producer.disconnect()
        train_producer.disconnect()
        evaluate_producer.disconnect()

    return {
        'filters': keys,
        'outputs': parameters['hyperparameters']
    }


def archive_optimization_outputs(
    filters: List[str],
    outputs: Dict[str, Any]
) -> Dict[str, Any]:
    """ Processes and stores all optimization outputs for subsequent use

    Args:
        filters (list(str)): Composite IDs of a federated combination
        outputs (dict): Outputs from a federated job
    Returns:
        Generated model (dict)      
    """
    created_run = run_records.create(*filters, details=outputs)

    return {'run': created_run}

#############
# Resources #
#############

@ns_api.route('/')
@ns_api.response(403, 'Optimizations not active')
@ns_api.response(404, 'Optimizations not found')
@ns_api.response(500, 'Internal failure')
class Optimizations(Resource):
    """ Handles hyperparameter tuning  within the PySyft grid. This targets the
        specific experimental model for optimization given a user-defined search
        space and performs a full federated cycle within the scope of 
        hyperparameter ranges.
    """

    @ns_api.doc("get_optimizations")
    @ns_api.marshal_with(payload_formatter.plural_model)
    def get(self, collab_id, project_id, expt_id):
        """ Retrieves global model corresponding to experiment and run 
            parameters for a specified project
        """
        ###########################
        # Implementation Footnote #
        ###########################

        # [Cause]
        # Optimizations trigger hyperparameter tuning, which requires running
        # multiple federated combinations concurrently. However, in Synergos
        # Basic, there exists only 1 grid.

        # [Problems]
        # When trying to run multiple federated combinations on a single grid,
        # Ray.Tune's parallelization causes problems in termination of an
        # existing grid (based on current implementation).

        # [Solution]
        # Limit hyperparameter tuning to only Synergos' SynCluster 
        # configuration, which supports multiple nodes, until further notice.

        if not is_cluster:
            logging.error(
                "Optimization operations only exist in SynCluster!",
                code=403,
                description="Optimization is only active in a Synergos cluster, and is unsupported in Synergos Basic.",
                ID_path=SOURCE_FILE,
                ID_class=Optimizations.__name__, 
                ID_function=Optimizations.get.__name__,
                **request.view_args
            )
            ns_api.abort(
                code=403, 
                message=f"Optimization is only active in a Synergos cluster, and is unsupported in Synergos Basic."
            )

        retrieved_validations = validation_records.read_all(
            filter=request.view_args
        )
        optim_validations = [
            record 
            for record in retrieved_validations
            if optim_prefix in record['key']['run_id']
        ]
        
        if optim_validations:
            
            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="optimizations.get",
                params=request.view_args,
                data=optim_validations
            )

            logging.info(
                f"Collaboration '{collab_id}' > Project '{project_id}' > Experiment '{expt_id}' > Optimizations: Record(s) retrieval successful!",
                code=200, 
                description="Optimization(s) specified federated conditions were successfully retrieved!",
                ID_path=SOURCE_FILE,
                ID_class=Optimizations.__name__, 
                ID_function=Optimizations.get.__name__,
                **request.view_args
            )

            return success_payload, 200

        else:
            logging.error(
                f"Collaboration '{collab_id}' > Project '{project_id}' -> Experiment '{expt_id}' -> Optimizations:  Record(s) retrieval failed.",
                code=404,
                description="Optimization statistics do not exist for specified keyword filters yet!",
                ID_path=SOURCE_FILE,
                ID_class=Optimizations.__name__, 
                ID_function=Optimizations.get.__name__,
                **request.view_args
            )
            ns_api.abort(
                code=404, 
                message=f"Optimizations do not exist for specified keyword filters!"
            )


    @ns_api.doc("trigger_optimizations")
    @ns_api.marshal_with(payload_formatter.plural_model)
    def post(self, collab_id, project_id, expt_id):
        """ Creates sets of hyperparameters using a specified AutoML algorithm,
            within the scope of a user-specified search space, and conducts a 
            federated cycle for each proposed set. A federated cycle involves:
            1) Training model using specified experimental architecture on
               training data across the grid
            2) Validating trained model on validation data across the grid

            JSON received is expected to contain the following information:

            eg.

            {
                'search_space': {
                    "batch_size": {"_type":"choice", "_value": [16, 32, 64, 128]},
                    "hidden_size":{"_type":"choice","_value":[128, 256, 512, 1024]},
                    "lr":{"_type":"choice","_value":[0.0001, 0.001, 0.01, 0.1]},
                    "momentum":{"_type":"uniform","_value":[0, 1]}
                },

                # Specify Backend type to use
                'backend': "nni",

                # Specify backend-related kwargs
                'tuner': "TPE",
                'metric': "accuracy",
                'optimize_mode': "maximize",
                'trial_concurrency': 1,
                'max_exec_duration': "1h",
                'max_trial_num': 10,
                'is_remote': True,
                'use_annotation': True,

                # Specify generic kwargs
                'auto_align': True,
                'dockerised': True,
                'verbose': True,
                'log_msgs': True
            }
        """
        ###########################
        # Implementation Footnote #
        ###########################

        # [Cause]
        # Optimizations trigger hyperparameter tuning, which requires running
        # multiple federated combinations concurrently. However, in Synergos
        # Basic, there exists only 1 grid.

        # [Problems]
        # When trying to run multiple federated combinations on a single grid,
        # Ray.Tune's parallelization causes problems in termination of an
        # existing grid (based on current implementation).

        # [Solution]
        # Limit hyperparameter tuning to only Synergos' SynCluster 
        # configuration, which supports multiple nodes, until further notice.

        if not is_cluster:
            logging.error(
                "Optimization operations only exist in SynCluster!",
                code=403,
                description="Optimization is only active in a Synergos cluster, and is unsupported in Synergos Basic.",
                ID_path=SOURCE_FILE,
                ID_class=Optimizations.__name__, 
                ID_function=Optimizations.get.__name__,
                **request.view_args
            )
            ns_api.abort(
                code=403, 
                message="Optimization is only active in a Synergos cluster, and is unsupported in Synergos Basic."
            )

        # Retrieve all connectivity settings for all Synergos components
        retrieved_collaboration = collab_records.read(collab_id=collab_id)

        queue_info = retrieved_collaboration['mq']
        queue_host = queue_info['host']
        queue_port = queue_info['ports']['main']

        # Retrieve project's ML action (eg. regression vs classification)
        retrieved_project = project_records.read(
            collab_id=collab_id,
            project_id=project_id
        )
        project_action = retrieved_project['action']

        # Retrieve specific experiment 
        retrieved_expt = expt_records.read(
            collab_id=collab_id,
            project_id=project_id,
            expt_id=expt_id
        )

        # Retrieve all deployed grids
        registrations = registration_records.read_all(
            filter={'collab_id': collab_id, 'project_id': project_id}
        )
        usable_grids = rpc_formatter.extract_grids(registrations)

        # Populate hyperparameter tuning parameters
        tuning_params = request.json

        # Create log directory
        optim_log_dir = os.path.join(
            out_dir, 
            collab_id, 
            project_id, 
            expt_id,
            "optimizations"
        )

        try:
            backend = tuning_params.get('backend', "tune")
            
            hypertuner = HYPERTUNER_BACKENDS[backend](
                host=queue_host,
                port=queue_port,
                log_dir=optim_log_dir
            )
            hypertuner.tune(
                keys=request.view_args,
                grids=usable_grids,
                action=project_action,
                experiment=retrieved_expt,
                **tuning_params
            )

        except KeyError:
            logging.error(
                "Collaboration '{}' > Project '{}' > Model '{}' > Optimizations: Record(s) creation failed.".format(
                    collab_id, project_id, expt_id
                ),
                code=417, 
                description="Inappropriate collaboration configurations passed!", 
                ID_path=SOURCE_FILE,
                ID_class=Optimizations.__name__, 
                ID_function=Optimizations.post.__name__,
                **request.view_args
            )
            ns_api.abort(                
                code=417,
                message=f"Specified backend '{backend}' is not supported!"
            )

        retrieved_runs = run_records.read_all(
            filter=request.view_args
        )
        optim_runs = [
            record 
            for record in retrieved_runs
            if optim_prefix in record['key']['run_id']
        ]

        if optim_runs:
                
            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="optimizations.post",
                params=request.view_args,
                data=optim_runs
            )

            logging.info(
                "Collaboration '{}' > Project '{}' > Model '{}' > Optimizations: Record(s) creation successful!".format(
                    collab_id, project_id, expt_id
                ),
                code=200, 
                description="Optimization(s) specified federated conditions were successfully retrieved!",
                ID_path=SOURCE_FILE,
                ID_class=Optimizations.__name__, 
                ID_function=Optimizations.get.__name__,
                **request.view_args
            )

            return success_payload, 200


        else:
            logging.error(
                "Collaboration '{}' > Project '{}' > Model '{}' > Optimizations: Record(s) creation failed.".format(
                    collab_id, project_id, expt_id
                ),
                code=404,
                description="Optimizations do not exist for specified keyword filters!",
                ID_path=SOURCE_FILE,
                ID_class=Optimizations.__name__, 
                ID_function=Optimizations.get.__name__,
                **request.view_args
            )
            ns_api.abort(
                code=404, 
                message=f"Optimizations do not exist for specified keyword filters!"
            ) 