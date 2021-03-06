#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in

# Libs
from flask import Blueprint
from flask_restx import Api

# Custom
from rest_rpc.evaluation.validations import ns_api as validation_ns
from rest_rpc.evaluation.predictions import ns_api as prediction_ns

##################
# Configurations #
##################

blueprint = Blueprint('evaluation', __name__)

api = Api(
    app=blueprint,
    version="1.0",
    title="Synergos Orchestrator REST-RPC Evaluation API", 
    description="API to facilitate model inference between TTP & participants"
)

#############################
# Validation management API #
#############################
"""
Supported routes:
1) "collaborations/<collab_id>/validations"
1) "collaborations/<collab_id>/validations/<project_id>/<expt_id>/<run_id>/<participant_id>"

1) "collaborations/<collab_id>/projects/<project_id>/validations"
2) "collaborations/<collab_id>/projects/<project_id>/validations/<expt_id>"
3) "collaborations/<collab_id>/projects/<project_id>/validations/<expt_id>/<run_id>"
4) "collaborations/<collab_id>/projects/<project_id>/validations/<expt_id>/<run_id>/<participant_id>"
"""

api.add_namespace(
    validation_ns,
    path="/collaborations/<collab_id>/projects/<project_id>/validations"
)

#############################
# Prediction management API #
#############################
"""
Supported routes:
1) "/participants/<participant_id>/collaborations/<collab_id>/predictions"
2) "/participants/<participant_id>/collaborations/<collab_id>/predictions/<project_id>"
3) "/participants/<participant_id>/collaborations/<collab_id>/predictions/<project_id>/<expt_id>"
4) "/participants/<participant_id>/collaborations/<collab_id>/predictions/<project_id>/<expt_id>/<run_id>"
"""

api.add_namespace(
    prediction_ns,
    path="/participants/<participant_id>/collaborations/<collab_id>/predictions"
)