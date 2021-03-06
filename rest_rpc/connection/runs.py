#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import os

# Libs
import jsonschema
from flask import request
from flask_restx import Namespace, Resource, fields

# Custom
from rest_rpc import app
from rest_rpc.connection.core.utils import TopicalPayload
from rest_rpc.training.models import model_output_model
from rest_rpc.evaluation.validations import val_output_model
from rest_rpc.evaluation.predictions import pred_output_model
from synarchive.connection import RunRecords

##################
# Configurations #
##################

SOURCE_FILE = os.path.abspath(__file__)

ns_api = Namespace(
    "runs", 
    description='API to faciliate run management in in a PySyft Grid.'
)

schemas = app.config['SCHEMAS']
db_path = app.config['DB_PATH']
run_records = RunRecords(db_path=db_path)

logging = app.config['NODE_LOGGER'].synlog
logging.debug("connection/runs.py logged", Description="No Changes")

###########################################################
# Models - Used for marshalling (i.e. moulding responses) #
###########################################################

config_model = ns_api.model(
    name="configurations",
    model={
        "input_size": fields.Integer(),
        "output_size": fields.Integer(),
        "is_condensed": fields.Boolean(),
        "rounds": fields.Integer(required=True),
        "epochs": fields.Integer(required=True),
        "batch_size": fields.Integer(),
        "lr": fields.Float(),
        "lr_decay": fields.Float(),
        "weight_decay": fields.Float(),
        "seed": fields.Integer(),
        "precision_fractional": fields.Integer(),
        "use_CLR": fields.Boolean(),
        "mu": fields.Float(),
        "reduction": fields.String(),
        "l1_lambda": fields.Float(),
        "l2_lambda": fields.Float(),
        "dampening": fields.Float(),
        "base_lr": fields.Float(),
        "max_lr": fields.Float(),
        "step_size_up": fields.Integer(),
        "step_size_down": fields.Integer(),
        "mode": fields.String(),
        "gamma": fields.Float(),
        "scale_mode": fields.String(),
        "cycle_momentum": fields.Boolean(),
        "base_momentum": fields.Float(),
        "max_momentum": fields.Float(),
        "last_epoch": fields.Integer(),
        "patience": fields.Integer(),
        "delta": fields.Float(),
        "cumulative_delta": fields.Boolean()
    }
)

run_input_model = ns_api.inherit(
    "run_input", 
    config_model, 
    {"run_id": fields.String()}
)

run_output_model = ns_api.inherit(
    "run_output",
    config_model,
    {
        'doc_id': fields.String(),
        'kind': fields.String(),
        'key': fields.Nested(
            ns_api.model(
                name='key',
                model={
                    'collab_id': fields.String(),
                    'project_id': fields.String(),
                    'expt_id': fields.String(),
                    'run_id': fields.String()
                }
            ),
            required=True
        ),
        'relations': fields.Nested(
            ns_api.model(
                name='run_relations',
                model={
                    'Model': fields.List(
                        fields.Nested(model_output_model, skip_none=True)
                    ),
                    'Validation': fields.List(
                        fields.Nested(val_output_model, skip_none=True)
                    ),
                    'Prediction': fields.List(
                        fields.Nested(pred_output_model, skip_none=True)
                    )
                }
            ),
            default={},
            required=True
        )
    }
)

payload_formatter = TopicalPayload(
    subject=run_records.subject, 
    namespace=ns_api, 
    model=run_output_model
)

#############
# Resources #
#############

@ns_api.route('/')
@ns_api.response(500, 'Internal failure')
class Runs(Resource):
    """ Handles the entire collection of runs as a catalogue """

    @ns_api.doc("get_runs")
    @ns_api.marshal_list_with(payload_formatter.plural_model)
    def get(self, collab_id, project_id, expt_id):
        """ Retrieve all run configurations queued for training """
        all_relevant_runs = run_records.read_all(
            filter={
                'collab_id': collab_id,
                'project_id': project_id, 
                'expt_id': expt_id
            }
        )

        success_payload = payload_formatter.construct_success_payload(
            status=200,
            method="runs.get",
            params=request.view_args,
            data=all_relevant_runs
        )

        logging.info(
            f"Collaboration '{collab_id}' -> Project '{project_id}' -> Experiment '{expt_id}' -> Runs: Bulk record retrieval successful!",
            code=200, 
            description=f"Runs under experiment '{expt_id}' of project '{project_id}' were successfully retrieved!", 
            ID_path=SOURCE_FILE,
            ID_class=Runs.__name__, 
            ID_function=Runs.get.__name__,
            **request.view_args
        )
        
        return success_payload, 200        


    @ns_api.doc("register_run")
    @ns_api.expect(run_input_model)
    @ns_api.marshal_with(payload_formatter.singular_model)
    @ns_api.response(201, "New run created!")
    @ns_api.response(417, "Inappropriate run configurations passed!")
    def post(self, collab_id, project_id, expt_id):
        """ Takes in a set of FL training run configurations and stores it """
        try:
            new_run_details = request.json
            run_id = new_run_details.pop('run_id')

            run_records.create(
                collab_id=collab_id,
                project_id=project_id, 
                expt_id=expt_id,
                run_id=run_id,
                details=new_run_details
            )
            retrieved_run = run_records.read(
                collab_id=collab_id,
                project_id=project_id, 
                expt_id=expt_id,
                run_id=run_id
            )

            success_payload = payload_formatter.construct_success_payload(
                status=201, 
                method="runs.post",
                params=request.view_args,
                data=retrieved_run
            )

            logging.info(
                f"Collaboration '{collab_id}' -> Project '{project_id}' -> Experiment '{expt_id}' -> Run '{run_id}': Record creation successful!", 
                description=f"Run '{run_id}' under experiment '{expt_id}' of project '{project_id}' was successfully created!", 
                code=201, 
                ID_path=SOURCE_FILE,
                ID_class=Runs.__name__, 
                ID_function=Runs.post.__name__,
                **request.view_args
            )
            
            return success_payload, 201

        except jsonschema.exceptions.ValidationError:
            logging.error(
                f"Collaboration '{collab_id}' -> Project '{project_id}' -> Experiment '{expt_id}' -> Run '{run_id}': Record creation failed.",
                code=417,
                description="Inappropriate run configurations passed!", 
                ID_path=SOURCE_FILE,
                ID_class=Runs.__name__, 
                ID_function=Runs.post.__name__,
                **request.view_args
            )
            ns_api.abort(
                code=417,
                message="Inappropriate run configurations passed!"
            )



@ns_api.route('/<run_id>')
@ns_api.response(404, 'Run not found')
@ns_api.response(500, 'Internal failure')
class Run(Resource):
    """ Handles all TTP interactions for managing run registration """
    
    @ns_api.doc("get_run")
    @ns_api.marshal_with(payload_formatter.singular_model)
    def get(self, collab_id, project_id, expt_id, run_id):
        """ Retrieves all runs registered for an experiment under a project """
        retrieved_run = run_records.read(
            collab_id=collab_id,
            project_id=project_id, 
            expt_id=expt_id,
            run_id=run_id
        )

        if retrieved_run:
            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="run.get",
                params=request.view_args,
                data=retrieved_run
            )

            logging.info(
                f"Collaboration '{collab_id}' -> Project '{project_id}' -> Experiment '{expt_id}' -> Run '{run_id}': Single record retrieval successful!", 
                code=200, 
                description=f"Run '{run_id}' under experiment '{expt_id}' of project '{project_id}' was successfully retrieved!", 
                ID_path=SOURCE_FILE,
                ID_class=Run.__name__, 
                ID_function=Run.get.__name__,
                **request.view_args
            )
            
            return success_payload, 200

        else:
            logging.error(
                f"Collaboration '{collab_id}' -> Project '{project_id}' -> Experiment '{expt_id}' -> Run '{run_id}': Single record retrieval failed!",
                code=404, 
                description=f"Run '{run_id}' does not exist for Experiment {expt_id} under Project '{project_id}'!",
                ID_path=SOURCE_FILE,
                ID_class=Run.__name__, 
                ID_function=Run.get.__name__,
                **request.view_args
            )
            ns_api.abort(
                code=404, 
                message=f"Run '{run_id}' does not exist for Experiment {expt_id} under Project '{project_id}'!"
            )


    @ns_api.doc("update_run")
    @ns_api.expect(config_model)
    @ns_api.marshal_with(payload_formatter.singular_model)
    def put(self, collab_id, project_id, expt_id, run_id):
        """ Updates a run's specified configurations IF & ONLY IF the run has
            yet to begin
        """
        try:
            run_updates = request.json

            run_records.update(
                collab_id=collab_id,
                project_id=project_id, 
                expt_id=expt_id,
                run_id=run_id,
                updates=run_updates
            )
            retrieved_run = run_records.read(
                collab_id=collab_id,
                project_id=project_id, 
                expt_id=expt_id,
                run_id=run_id
            )

            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="run.put",
                params=request.view_args,
                data=retrieved_run
            )

            logging.info(
                f"Collaboration '{collab_id}' -> Project '{project_id}' -> Experiment '{expt_id}' -> Run '{run_id}': Record update successful!",
                code=200,
                description=f"Run '{run_id}' under experiment '{expt_id}' of project '{project_id}' was successfully updated!", 
                ID_path=SOURCE_FILE,
                ID_class=Run.__name__, 
                ID_function=Run.put.__name__,
                **request.view_args
            )

            return success_payload, 200

        except jsonschema.exceptions.ValidationError:
            logging.error(
                f"Collaboration '{collab_id}' -> Project '{project_id}' -> Experiment '{expt_id}' -> Run '{run_id}': Record update failed.",
                code=417, 
                description="Inappropriate run configurations passed!", 
                ID_path=SOURCE_FILE,
                ID_class=Run.__name__, 
                ID_function=Run.put.__name__,
                **request.view_args
            )
            ns_api.abort(                
                code=417,
                message="Inappropriate experimental configurations passed!"
            )


    @ns_api.doc("delete_run")
    @ns_api.marshal_with(payload_formatter.singular_model)
    def delete(self, collab_id, project_id, expt_id, run_id):
        """ De-registers a previously registered run and deletes it """
        retrieved_run = run_records.read(
            collab_id=collab_id,
            project_id=project_id, 
            expt_id=expt_id,
            run_id=run_id
        )
        deleted_run = run_records.delete(
            collab_id=collab_id,
            project_id=project_id,
            expt_id=expt_id,
            run_id=run_id
        )

        if deleted_run:
            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="run.delete",
                params=request.view_args,
                data=retrieved_run
            )
            logging.info(
                f"Collaboration '{collab_id}' -> Project '{project_id}' -> Experiment '{expt_id}' -> Run '{run_id}': Record deletion successful!",
                code=200, 
                description=f"Run '{run_id}' under experiment '{expt_id}' of project '{project_id}' was successfully deleted!", 
                ID_path=SOURCE_FILE,
                ID_class=Run.__name__, 
                ID_function=Run.delete.__name__,
                **request.view_args
            )
            return success_payload, 200

        else:
            logging.error(
                f"Collaboration '{collab_id}' -> Project '{project_id}' -> Experiment '{expt_id}' -> Run '{run_id}': Record deletion failed.", 
                code=404, 
                description=f"Run '{run_id}' under experiment '{expt_id}' of project '{project_id}' does not exist!", 
                ID_path=SOURCE_FILE,
                ID_class=Run.__name__, 
                ID_function=Run.delete.__name__,
                **request.view_args
            )
            ns_api.abort(
                code=404, 
                message=f"Run '{run_id}' does not exist in for Experiment {expt_id} under Project '{project_id}'!"
            )
            