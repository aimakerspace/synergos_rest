#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
from types import ModuleType
from typing import Union, Callable

# Libs
from flask import Flask

# Custom


##################
# Configurations #
##################

app = Flask(__name__)

###########
# Helpers #
###########

def initialize_app(settings: Union[ModuleType, Callable, str]) -> Flask:
    """ Takes in a set of configurations to configure REST-RPC server
        operations. 
        
        Settings declared can take the following forms:
        1. Module (eg. import config -> app = initialize_app(settings=config))
        2. Class  (eg. class Config -> initialize_app(settings=config))
        3. Import string (eg. initialize_app(settings="package.config")        
        
        In all setting instances, all essential parameters MUST be 
        defined. Essential parameters are as follows:
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

    Args:
        settings (ModuleType/Callable/str): Module configurations to calibrate
            REST-RPC operations
    Returns:
        Initialized Flask app (Flask)
    """
    app.config.from_object(settings)
    
    from .connection import blueprint as connection_api
    from .training import blueprint as training_api
    from .evaluation import blueprint as evaluation_api

    app.register_blueprint(connection_api, url_prefix='/ttp/connect')
    app.register_blueprint(training_api, url_prefix='/ttp/train')
    app.register_blueprint(evaluation_api, url_prefix='/ttp/evaluate')

    return app