#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
from typing import Dict, Any

# Libs
from flask import Flask

# Custom
import config as defaults

##################
# Configurations #
##################

app = Flask(__name__)

# Load in default settings
app.config.from_object(defaults)

###########
# Helpers #
###########

def configure_app(config_file: str = None) -> Flask:
    """ Given a set of specified configurations, initialize Synergos REST-RPC
        app for orchestration.
        Note:
        1) All parameters MUST be in UPPERCASE (eg. GRID = 0)
        2) All essential parameters MUST be defined. Essential parameters are
           as follows:
           1.  GRID
           2.  IS_MASTER
           3.  IN_DIR
           4.  OUT_DIR
           5.  DATA_DIR
           6.  TEST_DIR
           7.  MLFLOW_DIR
           8.  CACHE
           9.  CORES_USED
           10. GPU_COUNT
           11. GPUS
           12. USE_GPU
           13. DEVICE
           14. RETRY_INTERVAL
           15. DB_PATH
           16. PAYLOAD_TEMPLATE
           17. NODE_LOGGER
           18. SYSMETRIC_LOGGER
            
    """
    if config_file:
        # Override defaults with customized configurations
        app.config.from_pyfile(config_file)

    return app


def initialize_app():
    """
    """
    from .connection import blueprint as connection_api
    from .training import blueprint as training_api
    from .evaluation import blueprint as evaluation_api

    app.register_blueprint(connection_api, url_prefix='/ttp/connect')
    app.register_blueprint(training_api, url_prefix='/ttp/train')
    app.register_blueprint(evaluation_api, url_prefix='/ttp/evaluate')

    return app