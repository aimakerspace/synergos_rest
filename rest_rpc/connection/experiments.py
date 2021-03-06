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
from rest_rpc.connection.runs import run_output_model
from rest_rpc.training.models import model_output_model
from rest_rpc.evaluation.validations import val_output_model
from rest_rpc.evaluation.predictions import pred_output_model
from synarchive.connection import ExperimentRecords

##################
# Configurations #
##################

SOURCE_FILE = os.path.abspath(__file__)

ns_api = Namespace(
    "experiments", 
    description='API to faciliate experiment management in a PySyft Grid.'
)

db_path = app.config['DB_PATH']
expt_records = ExperimentRecords(db_path=db_path)

logging = app.config['NODE_LOGGER'].synlog
logging.debug("connection/experiments.py logged", Description="No Changes")

###########################################################
# Models - Used for marshalling (i.e. moulding responses) #
###########################################################

# Note: In Flask-restx==0.2.0, 
# Creating a marshallable model from a specified JSON schema is bugged. While it
# is possible to use a schema model for formatting expectations, it cannot be
# used for marshalling outputs.
# Error thrown -> AttributeError: 'SchemaModel' object has no attribute 'items'
# Mitigation   -> Manually implement schema model until bug is fixed
""" 
[REDACTED in Flask-restx==0.2.0]
expt_schema = app.config['SCHEMAS']['experiment_schema']
structure_model = ns_api.schema_model(name='structure', schema=expt_schema)
"""
class ListableInteger():
    def format(self, value):
        return value

structure_model = ns_api.model(
    name='structure',
    model={
        "activation": fields.String(),
        "add_bias_kv": fields.Boolean(),
        "add_zero_attn": fields.Boolean(),
        "affine": fields.Boolean(),
        "align_corners": fields.Boolean(),
        "alpha": fields.Float(),
        "batch_first": fields.Boolean(),
        "beta": fields.Float(),
        "bias": fields.Boolean(),
        "bidirectional": fields.Boolean(),
        "blank": fields.Integer(),
        "ceil_mode": fields.Boolean(),
        "count_include_pad": fields.Boolean(),
        "cutoffs": fields.List(fields.String()),
        "d_model": fields.Integer(),
        "device_ids": fields.List(fields.Integer()),
        "dilation": fields.Integer(),
        "dim": fields.Integer(),
        "dim_feedforward": fields.Integer(),
        "div_value": fields.Float(),
        # "divisor_override": {
        #     "description": "if specified, it will be used as divisor in place of kernel_size"
        # },
        "dropout": fields.Float(),
        "elementwise_affine": fields.Boolean(),
        "embed_dim": fields.Integer(),
        "embedding_dim": fields.Integer(),
        "end_dim": fields.Integer(),
        "eps": fields.Float(),
        "full": fields.Boolean(),
        "groups": fields.Integer(),
        "head_bias": fields.Boolean(),
        "hidden_size": fields.Integer(),        # Flagged for arrayable value,
        "ignore_index": fields.Integer(),
        "in1_features": fields.Integer(),
        "in2_features": fields.Integer(),
        "in_channels": fields.Integer(),
        "in_features": fields.Integer(),
        "init": fields.Float(),
        "inplace": fields.Boolean(),
        "input_size": fields.Integer(),
        "k": fields.Float(),
        "kdim": fields.Integer(),
        "keepdim": fields.Boolean(),
        "kernel_size": fields.Integer(),        # Flagged for arrayable value
        "lambd": fields.Float(),
        "log_input": fields.Boolean(),
        "lower": fields.Float(),
        "margin": fields.Float(),
        "max_norm": fields.Float(),
        "max_val": fields.Float(),
        "min_val": fields.Float(),
        "mode": fields.String(),
        "momentum": fields.Float(),
        "n_classes": fields.Integer(),
        "negative_slope": fields.Float(),
        "nhead": fields.Integer(),
        "nonlinearity": fields.String(),
        "norm_type": fields.Float(),
        "normalized_shape": fields.Integer(),   # Flagged for arrayable value
        "num_channels": fields.Integer(),
        "num_chunks": fields.Integer(),
        "num_decoder_layers": fields.Integer(),
        "num_embeddings": fields.Integer(),
        "num_encoder_layers": fields.Integer(),
        "num_features": fields.Integer(),
        "num_groups": fields.Integer(),
        "num_heads": fields.Integer(),
        "num_layers": fields.Integer(),
        "num_parameters": fields.Integer(),
        "out_channels": fields.Integer(),
        "out_features": fields.Integer(),
        "output_device": fields.Integer(),
        "output_padding": fields.Integer(),
        "output_ratio": fields.Float(),         # Flagged for arrayable value
        "output_size": fields.Integer(),        # Flagged for arrayable value
        "p": fields.Float(),
        "padding": fields.Integer(),
        "padding_idx": fields.Integer(),
        "padding_mode": fields.String(),
        "pos_weight": fields.List(fields.Float()),
        "reduction": fields.String(),
        "requires_grad": fields.Boolean(),
        "return_indices": fields.Boolean(),
        "scale_factor": fields.Float(),         # Flagged for arrayable value
        "scale_grad_by_freq": fields.Boolean(),
        "size": fields.Integer(),               # Flagged for arrayable value
        "size_average": fields.Boolean(),
        "sparse": fields.Boolean(),
        "start_dim": fields.Integer(),
        "stride": fields.Integer(),             # Flagged for arrayable value
        "swap": fields.Boolean(),
        "threshold": fields.Float(),
        "track_running_stats": fields.Boolean(),
        "upper": fields.Float(),
        "upscale_factor": fields.Integer(),
        "value": fields.Float(),
        "vdim": fields.Integer(),
        "zero_infinity": fields.Boolean()
    }
)

layer_model = ns_api.model(
    name="layer",
    model={
        'is_input': fields.Boolean(required=True),
        'structure': fields.Nested(
            model=structure_model, 
            skip_none=True,
            required=True
        ),
        'l_type': fields.String(required=True),
        'activation': fields.String(required=True)
    }
)

expt_model = ns_api.model(
    name="experiment",
    model={
        'model': fields.List(
            fields.Nested(layer_model, required=True, skip_none=True)
        )
    }
)

expt_input_model = ns_api.inherit(
    "experiment_input",
    expt_model,
    {'expt_id': fields.String()}
)

expt_output_model = ns_api.inherit(
    "experiment_output",
    expt_model,
    {
        'doc_id': fields.String(),
        'kind': fields.String(),
        'key': fields.Nested(
            ns_api.model(
                name='key',
                model={
                    'collab_id': fields.String(),
                    'project_id': fields.String(),
                    'expt_id': fields.String()
                }
            ),
            required=True
        ),
        'relations': fields.Nested(
            ns_api.model(
                name='expt_relations',
                model={
                    'Run': fields.List(
                        fields.Nested(run_output_model, skip_none=True)
                    ),
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
    subject=expt_records.subject, 
    namespace=ns_api, 
    model=expt_output_model
)

#############
# Resources #
#############

@ns_api.route('/')
@ns_api.response(500, 'Internal failure')
class Experiments(Resource):
    """ Handles the entire collection of experiments as a catalogue """

    @ns_api.doc("get_experiments")
    @ns_api.marshal_list_with(payload_formatter.plural_model)
    def get(self, collab_id, project_id):
        """ Retrieve all run configurations queued for training """
        all_relevant_expts = expt_records.read_all(
            filter={
                'collab_id': collab_id, 
                'project_id': project_id
            }
        )

        success_payload = payload_formatter.construct_success_payload(
            status=200,
            method="experiments.get",
            params=request.view_args,
            data=all_relevant_expts
        )

        logging.info(
            f"Collaboration '{collab_id}' -> Project '{project_id}' -> Experiments: Bulk record retrieval successful!",
            code=200, 
            description=f"Experiments under project '{project_id}' were successfully retrieved!", 
            ID_path=SOURCE_FILE,
            ID_class=Experiments.__name__, 
            ID_function=Experiments.get.__name__,
            **request.view_args
        )

        return success_payload, 200


    @ns_api.doc("register_experiment")
    @ns_api.expect(expt_input_model)
    # @ns_api.marshal_with(payload_formatter.singular_model)
    @ns_api.response(201, "New experiment created!")
    @ns_api.response(417, "Inappropriate experiment configurations passed!")
    def post(self, collab_id, project_id):
        """ Takes a model configuration to be queued for training and stores it
        """
        try:
            new_expt_details = request.json
            expt_id = new_expt_details.pop('expt_id')

            expt_records.create(
                collab_id=collab_id,
                project_id=project_id, 
                expt_id=expt_id,
                details=new_expt_details
            )
            retrieved_expt = expt_records.read(
                collab_id=collab_id,
                project_id=project_id, 
                expt_id=expt_id
            )

            logging.debug(
                f"Created experiment tracked. Expt_ID: '{expt_id}'", 
                experiment=retrieved_expt, 
                ID_path=SOURCE_FILE,
                ID_class=Experiments.__name__, 
                ID_function=Experiments.post.__name__,
                **request.view_args
            )

            success_payload = payload_formatter.construct_success_payload(
                status=201, 
                method="experiments.post",
                params=request.view_args,
                data=retrieved_expt
            )
            
            logging.info(
                f"Collaboration '{collab_id}' -> Project '{project_id}' -> Experiment '{expt_id}': Record creation successful!", 
                description=f"Experiment '{expt_id}' was successfully submitted under project '{project_id}'!",
                code=201, 
                ID_path=SOURCE_FILE,
                ID_class=Experiments.__name__, 
                ID_function=Experiments.post.__name__,
                **request.view_args
            )

            return success_payload, 201

        except jsonschema.exceptions.ValidationError:
            logging.error(
                f"Experiment '{expt_id}': Record creation failed.",
                code=417,
                description="Inappropriate experimental configurations passed!", 
                ID_path=SOURCE_FILE,
                ID_class=Experiments.__name__, 
                ID_function=Experiments.post.__name__,
                **request.view_args
            )
            ns_api.abort(
                code=417,
                message="Inappropriate experiment configurations passed!"
            )



@ns_api.route('/<expt_id>')
@ns_api.response(404, 'Experiment not found')
@ns_api.response(500, 'Internal failure')
class Experiment(Resource):
    """ Handles all TTP interactions for managing experimental configuration.
        Such interactions involve listing, specifying, updating and cancelling 
        experiments.
    """

    @ns_api.doc("get_experiment")
    @ns_api.marshal_with(payload_formatter.singular_model)
    def get(self, collab_id, project_id, expt_id):
        """ Retrieves all experimental parameters corresponding to a specified
            project
        """
        retrieved_expt = expt_records.read(
            collab_id=collab_id,
            project_id=project_id, 
            expt_id=expt_id
        )

        if retrieved_expt:
            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="experiment.get",
                params=request.view_args,
                data=retrieved_expt
            )

            logging.info(
                f"Collaboration '{collab_id}' -> Project '{project_id}' -> Experiment '{expt_id}': Single record retrieval successful!", 
                code=200, 
                description=f"Experiment '{expt_id}' under project '{project_id}' was successfully retrieved!",
                ID_path=SOURCE_FILE,
                ID_class=Experiment.__name__, 
                ID_function=Experiment.get.__name__,
                **request.view_args
            )
            
            return success_payload, 200

        else:
            logging.error(
                f"Collaboration '{collab_id}' -> Project '{project_id}' -> Experiment '{expt_id}': Single record retrieval failed!",
                code=404, 
                description=f"Experiment '{expt_id}' does not exist in Project '{project_id}'!", 
                ID_path=SOURCE_FILE,
                ID_class=Experiment.__name__, 
                ID_function=Experiment.get.__name__,
                **request.view_args
            )
            ns_api.abort(
                code=404, 
                message=f"Experiment '{expt_id}' does not exist in Project '{project_id}'!"
            )
            

    @ns_api.doc("update_experiment")
    @ns_api.expect(expt_model)
    @ns_api.marshal_with(payload_formatter.singular_model)
    def put(self, collab_id, project_id, expt_id):
        """ Updates a participant's specified choices IF & ONLY IF his/her
            registered experiments have not yet commenced
        """
        try:
            expt_updates = request.json

            expt_records.update(
                collab_id=collab_id,
                project_id=project_id, 
                expt_id=expt_id,
                updates=expt_updates
            )
            retrieved_expt = expt_records.read(
                collab_id=collab_id,
                project_id=project_id, 
                expt_id=expt_id
            )

            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="experiment.put",
                params=request.view_args,
                data=retrieved_expt
            )

            logging.info(
                f"Collaboration '{collab_id}' -> Project '{project_id}' -> Experiment '{expt_id}': Record update successful!",
                code=200,
                description=f"Experiment '{expt_id}' under project '{project_id}' was successfully updated!", 
                ID_path=SOURCE_FILE,
                ID_class=Experiment.__name__, 
                ID_function=Experiment.put.__name__,
                **request.view_args
            )
            
            return success_payload, 200

        except jsonschema.exceptions.ValidationError:
            logging.error(
                f"Collaboration '{collab_id}' -> Project '{project_id}' -> Experiment '{expt_id}': Record update failed.",
                code=417, 
                description="Inappropriate experimental configurations passed!", 
                ID_path=SOURCE_FILE,
                ID_class=Experiment.__name__, 
                ID_function=Experiment.put.__name__,
                **request.view_args
            )
            ns_api.abort(                
                code=417,
                message="Inappropriate experimental configurations passed!"
            )
 

    @ns_api.doc("delete_experiment")
    @ns_api.marshal_with(payload_formatter.singular_model)
    def delete(self, collab_id, project_id, expt_id):
        """ De-registers previously registered experiment, and clears out all 
            metadata
        """
        retrieved_expt = expt_records.read(
            collab_id=collab_id,
            project_id=project_id, 
            expt_id=expt_id
        )
        deleted_expt = expt_records.delete(
            collab_id=collab_id,
            project_id=project_id,
            expt_id=expt_id
        )

        if deleted_expt:

            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="experiment.delete",
                params=request.view_args,
                data=retrieved_expt
            )
            logging.info(
                f"Collaboration '{collab_id}' -> Project '{project_id}' -> Experiment '{expt_id}': Record deletion successful!",
                code=200, 
                description=f"Experiment '{expt_id}' under project '{project_id}' was successfully deleted!",
                ID_path=SOURCE_FILE,
                ID_class=Experiment.__name__, 
                ID_function=Experiment.delete.__name__,
                **request.view_args
            )
            return success_payload

        else:
            logging.error(
                f"Collaboration '{collab_id}' -> Project '{project_id}' -> Experiment '{expt_id}': Record deletion failed.", 
                code=404, 
                description=f"Experiment '{expt_id}' does not exist in Project '{project_id}'!", 
                ID_path=SOURCE_FILE,
                ID_class=Experiment.__name__, 
                ID_function=Experiment.delete.__name__,
                **request.view_args
            )
            ns_api.abort(
                code=404, 
                message=f"Experiment '{expt_id}' does not exist in Project '{project_id}'!"
            )
