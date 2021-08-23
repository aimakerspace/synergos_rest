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
from rest_rpc.connection.tags import tag_output_model
from synarchive.connection import RegistrationRecords

##################
# Configurations #
##################

SOURCE_FILE = os.path.abspath(__file__)

ns_api = Namespace(
    "registration", 
    description='API to faciliate registation management in a PySyft Grid.'
)

schemas = app.config['SCHEMAS']
db_path = app.config['DB_PATH']
registration_records = RegistrationRecords(db_path=db_path)

logging = app.config['NODE_LOGGER'].synlog
logging.debug("connection/registration.py logged", Description="No Changes")

###########################################################
# Models - Used for marshalling (i.e. moulding responses) #
###########################################################

base_ports_model = ns_api.model(
    name="base_ports",
    model={
        'main': fields.Integer(),
        'ui': fields.Integer()
    }
)

base_connection_model = ns_api.model(
    name="connection",
    model={
        'host': fields.String(),
        'ports': fields.Nested(model=base_ports_model, allow_null=True),
        'secure': fields.Boolean()
    }
)

logger_ports_model = ns_api.inherit(
    "logger_ports",
    base_ports_model,
    {
        'sysmetrics': fields.Integer(),
        'director': fields.Integer(),
        'ttp': fields.Integer(),
        'worker': fields.Integer(),
    }
)

logs_model =  ns_api.model(
    name="logs_connection",
    model={
        'host': fields.String(),
        'ports': fields.Nested(model=logger_ports_model, allow_null=True),
        'secure': fields.Boolean()
    }
)

channel_model = ns_api.model(
    name="channel",
    model={
        'host': fields.String(required=True),
        'port': fields.Integer(required=True),
        'log_msgs': fields.Boolean(),
        'verbose': fields.Boolean(),
        'f_port': fields.Integer(required=True)
    }
)

connectivity_field = fields.Wildcard(fields.Nested(channel_model))
connectivity_model = ns_api.model(
    name="registered_connections",
    model={
        'role': fields.String(required=True),
        'n_count': fields.Integer(required=True),
        'node_*': connectivity_field
    }
)

incentives_field = fields.Wildcard(fields.Raw()) # tentative
incentive_model = ns_api.model(
    name="registered_incentives",
    model={"*": incentives_field}
)

social_fields = fields.Wildcard(fields.String())
social_model = ns_api.model(name="socials", model={"*": social_fields})

registration_model = ns_api.inherit(
    "registration",
    connectivity_model,
    {
        'collaboration': fields.Nested(
            ns_api.model(
                name="registered_collaboration",
                model={
                    # Catalogue Connection
                    'catalogue': fields.Nested(model=base_connection_model, skip_none=True),
                    # Logger Connection
                    'logs': fields.Nested(model=logs_model, skip_none=True),
                    # Meter Connection
                    'meter': fields.Nested(model=base_connection_model, skip_none=True),
                    # MLOps Connection
                    'mlops': fields.Nested(model=base_connection_model, skip_none=True),
                    # MQ Connection
                    'mq': fields.Nested(model=base_connection_model, skip_none=True)
                }
            ),
            required=True
        ),

        'project': fields.Nested(
            ns_api.model(
                name="registered_project",
                model={
                    'action': fields.String(required=True),
                    'incentives': fields.Nested(
                        model=incentive_model,
                        required=True,
                        skip_none=True
                    ),
                    'start_at': fields.String()
                }
            ),
            required=True
        ),

        'participant': fields.Nested(
            ns_api.model(
                name="registered_participant",
                model={
                    'id': fields.String(required=True),
                    'category': fields.List(fields.String(), default=[]),
                    'summary': fields.String(),
                    'phone': fields.String(),
                    'email': fields.String(),
                    'socials': fields.Nested(social_model, skip_none=True)
                }
            ),
            required=True
        )
    }
)

# No need for input model since no additional input from used required to
# generate a registration_id.

registration_output_model = ns_api.inherit(
    "registration_input",
    registration_model,
    {
        'doc_id': fields.String(),
        'kind': fields.String(),
        'key': fields.Nested(
            ns_api.model(
                name='key',
                model={
                    'collab_id': fields.String(),
                    'project_id': fields.String(),
                    'participant_id': fields.String()
                }
            ),
            required=True
        ),
        'relations': fields.Nested(
            ns_api.model(
                name='registration_relations',
                model={
                    'Tag': fields.List(
                        fields.Nested(tag_output_model, skip_none=True)
                    )
                }
            ),
            default={},
            required=True
        )
    }
)

# registration_export_model = ns_api.inherit(
#     "registration_export",
#     role_model,
#     {
#         'doc_id': fields.String(),
#         'kind': fields.String(),
#         'key': fields.Nested(
#             ns_api.model(
#                 name='key',
#                 model={
#                     'project_id': fields.String(),
#                     'participant_id': fields.String()
#                 }
#             ),
#             required=True
#         )
#     }
# )

payload_formatter = TopicalPayload(
    subject=registration_records.subject, 
    namespace=ns_api, 
    model=registration_output_model
)

#############
# Resources #
#############

@ns_api.response(500, 'Internal failure')
class Registrations(Resource):
    """ Handles the entire collection of projects as a catalogue """

    @ns_api.doc('list_registrations')
    @ns_api.marshal_list_with(payload_formatter.plural_model)
    def get(self, **kwargs):
        """ Retrieve all metadata for each set of registrations """
        all_relevant_registration = registration_records.read_all(filter=kwargs)

        success_payload = payload_formatter.construct_success_payload(
            status=200,
            method="registrations.get",
            params={},
            data=all_relevant_registration
        )

        logging.info(
            f"Registrations: Bulk record retrieval successful!",
            code=200, 
            description=f"All existing registrations were successfully retrieved!", 
            ID_path=SOURCE_FILE,
            ID_class=Registrations.__name__, 
            ID_function=Registrations.get.__name__,
            **kwargs
        )

        return success_payload, 200



@ns_api.response(404, 'Registration not found')
@ns_api.response(500, 'Internal failure')
class Registration(Resource):
    """ Handles all TTP & participant interactions for activity registration and
        role definition
    """

    @ns_api.doc('get_registration')
    @ns_api.marshal_with(payload_formatter.singular_model)
    def get(self, collab_id, project_id, participant_id):
        """ Retrieves all metadata describing specified project """
        retrieved_registration = registration_records.read(
            collab_id=collab_id,
            project_id=project_id,
            participant_id=participant_id
        )

        if retrieved_registration:
            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="registration.get",
                params=request.view_args,
                data=retrieved_registration
            )

            logging.info(
                f"Participant '{participant_id}' >|< Collaboration '{collab_id}' -> Project '{project_id}' -> Registration: Single record retrieval successful!", 
                code=200, 
                description=f"Registration of '{participant_id}' under project '{project_id}' was successfully retrieved!",
                ID_path=SOURCE_FILE,
                ID_class=Registration.__name__, 
                ID_function=Registration.get.__name__,
                **request.view_args
            )

            return success_payload, 200

        else:
            logging.error(
                f"Participant '{participant_id}' >|< Collaboration '{collab_id}' -> Project '{project_id}' -> Registration: Single record retrieval failed!",
                code=404, 
                description=f"Participant '{participant_id}' has not registered for Project '{project_id}'!",
                ID_path=SOURCE_FILE,
                ID_class=Registration.__name__, 
                ID_function=Registration.get.__name__,
                **request.view_args
            )
            ns_api.abort(
                code=404, 
                message=f"Participant '{participant_id}' has not registered for Project '{project_id}'!"
            )


    @ns_api.doc('register_registration')
    @ns_api.expect(connectivity_model) # for guiding payload
    @ns_api.marshal_with(payload_formatter.singular_model)
    @ns_api.response(201, "New registration entry created!")
    @ns_api.response(417, "Insufficient registration configurations passed!")
    def post(self, collab_id, project_id, participant_id):
        """ Takes in configurations for a participant to complete registration 
            for a project, which includes its supposed role in the training grid
            and stores it for use in orchestration
        """
        # try:
        new_registration_details = request.json

        registration_records.create(
            collab_id=collab_id,
            project_id=project_id, 
            participant_id=participant_id,
            details=new_registration_details
        )
        retrieved_registration = registration_records.read(
            collab_id=collab_id,
            project_id=project_id,
            participant_id=participant_id
        )

        success_payload = payload_formatter.construct_success_payload(
            status=201, 
            method="registration.post",
            params=request.view_args,
            data=retrieved_registration
        )

        logging.info(
            f"Participant '{participant_id}' >|< Collaboration '{collab_id}' -> Project '{project_id}' -> Registration: Record creation successful!", 
            description=f"Registration of '{participant_id}' under project '{project_id}' was successfully submitted!",
            code=201, 
            ID_path=SOURCE_FILE,
            ID_class=Registration.__name__, 
            ID_function=Registration.post.__name__,
            **request.view_args
        )
        
        return success_payload, 201

        # except jsonschema.exceptions.ValidationError:
        #     logging.error(
        #         f"Participant '{participant_id}' >|< Collaboration '{collab_id}' -> Project '{project_id}' -> Registration: Record creation failed.",
        #         code=417,
        #         description="Inappropriate registration configurations passed!", 
        #         ID_path=SOURCE_FILE,
        #         ID_class=Registration.__name__, 
        #         ID_function=Registration.post.__name__,
        #         **request.view_args
        #     )
        #     ns_api.abort(
        #         code=417,
        #         message="Inappropriate registration configurations passed!"
        #     )


    @ns_api.doc('update_registration')
    @ns_api.expect(connectivity_model)
    @ns_api.marshal_with(payload_formatter.singular_model)
    def put(self, collab_id, project_id, participant_id):
        """ Updates a participant's specified choices IF & ONLY IF his/her
            registered experiments have not yet commenced
        """
        try:
            registration_updates = request.json

            registration_records.update(
                collab_id=collab_id,
                project_id=project_id,
                participant_id=participant_id,
                updates=registration_updates
            )
            retrieved_registration = registration_records.read(
                collab_id=collab_id,
                project_id=project_id,
                participant_id=participant_id
            )

            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="registration.put",
                params=request.view_args,
                data=retrieved_registration
            )

            logging.info(
                f"Participant '{participant_id}' >|< Collaboration '{collab_id}' -> Project '{project_id}' -> Registration: Record update successful!",
                code=200,
                description=f"Registration of '{participant_id}' under project '{project_id}' was successfully updated!", 
                ID_path=SOURCE_FILE,
                ID_class=Registration.__name__, 
                ID_function=Registration.put.__name__,
                **request.view_args
            )
            
            return success_payload, 200

        except jsonschema.exceptions.ValidationError:
            logging.error(
                f"Participant '{participant_id}' >|< Collaboration '{collab_id}' -> Project '{project_id}' -> Registration: Record update failed.",
                code=417, 
                description="Inappropriate registration configurations passed!", 
                ID_path=SOURCE_FILE,
                ID_class=Registration.__name__, 
                ID_function=Registration.put.__name__,
                **request.view_args
            )
            ns_api.abort(                
                code=417,
                message="Inappropriate registration configurations passed!"
            )


    @ns_api.doc('delete_registration')
    @ns_api.marshal_with(payload_formatter.singular_model)
    def delete(self, collab_id, project_id, participant_id):
        """ De-registers participant from project and removes all their metadata
        """
        retrieved_registration = registration_records.read(
            collab_id=collab_id,
            project_id=project_id,
            participant_id=participant_id
        )
        deleted_registration = registration_records.delete(
            collab_id=collab_id,
            project_id=project_id,
            participant_id=participant_id
        )
        
        if deleted_registration:

            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="registration.delete",
                params=request.view_args,
                data=retrieved_registration
            )

            logging.info(
                f"Participant '{participant_id}' >|< Collaboration '{collab_id}' -> Project '{project_id}' -> Registration: Record deletion successful!",
                code=200, 
                description=f"Registration of '{participant_id}' under project '{project_id}' was successfully deleted!", 
                ID_path=SOURCE_FILE,
                ID_class=Registration.__name__, 
                ID_function=Registration.delete.__name__,
                **request.view_args
            )

            return success_payload

        else:
            logging.error(
                f"Participant '{participant_id}' >|< Collaboration '{collab_id}' -> Project '{project_id}' -> Registration: Record deletion failed.", 
                code=404, 
                description=f"Participant '{participant_id}' has not registered for Project '{project_id}'!",  
                ID_path=SOURCE_FILE,
                ID_class=Registration.__name__, 
                ID_function=Registration.delete.__name__,
                **request.view_args
            )
            ns_api.abort(
                code=404, 
                message=f"Participant '{participant_id}' has not registered for Project '{project_id}'!"
            )
            