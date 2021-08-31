#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Any

# Libs
import aiohttp
import mlflow
import torch as th

# Custom
from rest_rpc import app
from rest_rpc.training.core.utils import (
    UrlConstructor, 
    RPCFormatter, 
    Orchestrator
)
from synarchive.connection import ExperimentRecords, RunRecords
from synarchive.training import ModelRecords
from synarchive.evaluation import MLFRecords

##################
# Configurations #
##################

SOURCE_FILE = os.path.abspath(__file__)

schemas = app.config['SCHEMAS']
db_path = app.config['DB_PATH']
out_dir = app.config['OUT_DIR']
mlflow_dir = app.config['MLFLOW_DIR']

logging = app.config['NODE_LOGGER'].synlog
logging.debug("evaluation/core/utils.py logged", Description="No Changes")

####################
# Helper Functions #
####################

def replicate_combination_key(expt_id, run_id):
    return str((expt_id, run_id))

############################################
# Inference Orchestration class - Analyser #
############################################

class Analyser(Orchestrator):
    """ 
    Takes in a list of minibatch IDs and sends them to worker nodes. Workers
    will use these IDs to reconstruct their aggregated test datasets with 
    prediction labels mapped appropriately.

    Attributes:
        inferences (dict(str, dict(str, dict(str, th.tensor)))
    """
    def __init__(
        self, 
        collab_id: str,
        project_id: str,
        expt_id: str, 
        run_id: str,
        inferences: dict,
        metas: list = ['train', 'evaluate', 'predict'],
        auto_align: bool = True
    ):
        super().__init__()

        self.metas = metas
        self.collab_id = collab_id
        self.project_id = project_id
        self.expt_id = expt_id
        self.run_id = run_id
        self.inferences = inferences
        self.auto_align = auto_align

    ###########
    # Helpers #
    ###########

    async def _poll_for_stats(
        self, 
        node_info: Dict[str, Any], 
        inferences: Dict[str, Dict[str, th.Tensor]]
    ):
        """ Parses a registration record for participant metadata, before
            submitting minibatch IDs of inference objects to corresponding 
            worker node's REST-RPC service for calculating descriptive
            statistics and prediction exports

        Args:
            reg_record (tinydb.database.Document): Participant-project details
            inferences (dict): List of dicts containing inference object IDs
        Returns:
            Statistics (dict)
        """
        _, _, participant_id = self.parse_keys(node_info)
 
        if not inferences:
            return {participant_id: {meta:{} for meta in self.metas}}

        # Construct destination url for interfacing with worker REST-RPC
        rest_connection = self.parse_rest_info(node_info)
        destination_constructor = UrlConstructor(**rest_connection)
        destination_url = destination_constructor.construct_predict_url(
            collab_id=self.collab_id,
            project_id=self.project_id,
            expt_id=self.expt_id,
            run_id=self.run_id
        )

        ml_action = self.parse_action(node_info)
        data_tags = self.parse_tags(node_info)
        data_alignments = self.parse_alignments(node_info, self.auto_align)

        payload = {
            'action': ml_action, 
            'tags': data_tags,
            'alignments': data_alignments,
            'inferences': inferences
        }

        # Trigger remote inference by posting alignments & ID mappings to 
        # `Predict` route in worker
        resp_inference_data, status_code = await self.instruct(
            command='post', 
            url=destination_url, 
            payload=payload
        )
        
        logging.debug(
            f"Participant '{participant_id}' >|< Project '{self.project_id}' -> Experiment '{self.expt_id}' -> Run '{self.run_id}': Polled statistics tracked.",
            description=f"Polled statistics for participant '{participant_id}' under project '{self.project_id}' using experiment '{self.expt_id}' and run '{self.run_id}' tracked.",
            resp_json=resp_inference_data,
            ID_path=SOURCE_FILE,
            ID_class=Analyser.__name__,
            ID_function=Analyser._poll_for_stats.__name__
        )

        # Extract the relevant expt-run results
        expt_run_key = replicate_combination_key(self.expt_id, self.run_id)
        metadata = resp_inference_data['results'][expt_run_key]

        # Filter by selected meta-datasets
        filtered_statistics = {
            meta: stats 
            for meta, stats in metadata.items()
            if meta in self.metas
        }

        return {participant_id: filtered_statistics}


    async def _collect_all_stats(self, grid: List[Dict[str, Any]]) -> dict:
        """ Asynchronous function to submit inference data to registered
            participant servers in return for remote performance statistics

        Args:
            grid (list(dict))): Registry of participants' node information
        Returns:
            All participants' statistics (dict)
        """

        ###########################
        # Implementation Footnote #
        ###########################

        # [Causes]
        # Node info retrieved is asynmetric as compared to input inferences.

        # [Problems]
        # Zipping causes misallocation of dataset inferences to the wrong
        # servers, resulting in a "ValueError: Found input variables with 
        # inconsistent numbers of samples: [N1, N2]", since prediction counts 
        # obtained are not aligned with label counts in the remote servers 

        # [Solution]
        # Pair things up via manual matching

        mapped_pairs = [
            (record, inferences)
            for worker_id, inferences in self.inferences.items()
            for record in grid
            if self.parse_syft_info(record).get('id') == worker_id
        ]

        logging.debug(f"Mapped pairs: {mapped_pairs}")

        all_statistics = {}
        for future in asyncio.as_completed(
            map(lambda args: self._poll_for_stats(*args), mapped_pairs)
        ):
            result = await future
            all_statistics.update(result)

        return all_statistics

    ##################
    # Core Functions #
    ##################

    def infer(self, grid: List[Dict[str, Any]]) -> dict:
        """ Wrapper function for triggering asychroneous remote inferencing of
            participant nodes

        Args:
            grid (list(dict))): Registry of participants' node information
        Returns:
            All participants' statistics (dict)
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            all_stats = loop.run_until_complete(
                self._collect_all_stats(grid=grid)
            )
        finally:
            loop.close()

        return all_stats



####################################
# MLFLow logging class - MLFlogger #
####################################

class MLFlogger:
    """ 
    Wrapper class around MLFlow to faciliate experiment & run registrations in
    the REST-RPC setting, where statistical logging is performed during 
    post-mortem analysis
    """
    def __init__(self, db_path: str = db_path, remote_uri: str = None):

        # Private attributes
        self.__rpc_formatter = RPCFormatter()
        self.__expt_records = ExperimentRecords(db_path=db_path)
        self.__run_records = RunRecords(db_path=db_path)
        self.__model_records = ModelRecords(db_path=db_path)
        self.__connector = "_>_"
        
        # Public attributes
        self.mlf_records = MLFRecords(db_path=db_path)
        self.db_path = db_path
        self.remote_uri = remote_uri

    ###########
    # Helpers #
    ###########

    def _generate_expt_name(
        self, 
        collab_id: str, 
        project_id: str, 
        expt_id: str
    ) -> str:
        """ Abstracts creation of a unique key to identify an MLFlow experiment. 
            This collapses the payload hierarchy so that only one MLFlow instance
            is required to host all runs registered under collaboration(s)  

        Args:
            collab_id (str): ID of collaboration
            project_id (str): ID of project
            expt_id (str): ID of experiment
        Returns:
            Experiment name (str)
        """
        return self.__connector.join([collab_id, project_id, expt_id])


    def _generate_run_name(
        self,         
        collab_id: str, 
        project_id: str, 
        expt_id: str,
        run_id: str
    ) -> str:
        """ Abstracts creation of a unique key to identify an MLFlow run.

        Args:
            collab_id (str): ID of collaboration
            project_id (str): ID of project
            expt_id (str): ID of experiment
            run_id (str): ID of run
        Returns:
            Run name (str)
        """
        return run_id   # keeping it simple

    
    def _generate_record_name(
        self,
        collab_id: str, 
        project_id: str, 
        expt_id: str,
        run_id: str = "",
        mlflow_type: str = "experiment"
    ):
        """ Abstracts creation of a unique key to partition experiments & runs
            internally within the context of MLFlow.

        Args:
            collab_id (str): ID of collaboration
            project_id (str): ID of project
            expt_id (str): ID of experiment
            run_id (str): ID of run
            mlflow_type (str): Type of entity to be represented in MLFlow
        Returns:
            Record name (str)
        """
        
        ###########################
        # Implementation Footnote #
        ###########################

        # [Causes]
        # In MLFlow, only experiments and runs are mappable/usable to the 
        # Synergos hierarchy. In order to simplify the deployment process, 
        # collaborations & projects are collapsed in as part of an MLFlow's 
        # experiment, allowing a single instance of MLFlow to host results &
        # analysis of results across the grid.

        # [Problems]
        # The collapsed hierarchy becomes ambigious due to the inability to
        # distinguish between a Synergos experiment and a Synergos run.

        # [Solution]
        # Use expt_id and run_id to create a unique composite key for use when
        # storing mapping details between MLFlow and Synergos.

        if mlflow_type == "experiment":
            return expt_id

        elif mlflow_type == "run":
            return self.__connector.join([expt_id, run_id])

        else:
            raise RuntimeError("Unsupported entity specified!")


    def retrieve_mlflow_experiment(
        self,
        collab_id: str, 
        project_id: str,
        expt_id: str,
    ) -> Dict[str, str]:
        """
        """
        expt_name = self._generate_expt_name(collab_id, project_id, expt_id)
        expt_record_name = self._generate_record_name(
            collab_id, project_id, expt_id,
            mlflow_type="experiment"
        )
        expt_mlflow_record = self.mlf_records.read(
            collaboration=collab_id,
            project=project_id, 
            record=expt_record_name,
            name=expt_name
        )
        return expt_mlflow_record


    def retrieve_mlflow_run(
        self,
        collab_id: str, 
        project_id: str,
        expt_id: str,
        run_id: str
    ) -> Dict[str, str]:
        """
        """
        run_name = self._generate_run_name(collab_id, project_id, expt_id, run_id)
        run_record_name = self._generate_record_name(
            collab_id, project_id, expt_id, run_id,
            mlflow_type="run"
        )
        run_mlflow_record = self.mlf_records.read(
            collaboration=collab_id,
            project=project_id, 
            record=run_record_name,
            name=run_name
        )
        return run_mlflow_record


    def initialise_mlflow_experiment(
        self, 
        collab_id: str, 
        project_id: str,
        expt_id: str,
        tracking_uri: str
    ) -> Dict[str, str]:
        """ Initialises an MLFlow experiment at the specified project URI

        Args:
            project_id (str): REST-RPC ID of specified project
            expt_id (str): REST-RPC ID of specified experiment
        Returns:
            MLFlow Experiment configuration (dict)
        """
        mlflow.set_tracking_uri(tracking_uri)

        expt_name = self._generate_expt_name(collab_id, project_id, expt_id)
        expt_record_name = self._generate_record_name(
            collab_id, project_id, expt_id,
            mlflow_type="experiment"
        )
        expt_mlflow_record = self.mlf_records.read(
            collaboration=collab_id,
            project=project_id, 
            record=expt_record_name,
            name=expt_name
        )

        # Check if MLFlow experiment has already been created. Create a 
        # MLFRecord if: 
        # 1) No previous records for this combination key was initialized
        # 2) Specified tracking URI has changed

        if not expt_mlflow_record:
            
            # Initialise MLFlow experiment
            mlflow_id = mlflow.create_experiment(name=expt_name)

            mlflow_details = {
                'collaboration': collab_id,
                'project': project_id,
                'record': expt_record_name,
                'name': expt_name,
                'mlflow_type': 'experiment',
                'mlflow_id': mlflow_id,
                'mlflow_uri': tracking_uri  # Overridable with remote > local
            }
            expt_mlflow_record = self.mlf_records.create(
                collaboration=collab_id,
                project=project_id,
                record=expt_record_name,
                name=expt_name, 
                details=mlflow_details
            )

        # Update URI if necessary
        if expt_mlflow_record.get('mlflow_uri') != tracking_uri:
            expt_mlflow_record = self.mlf_records.update(
                collaboration=collab_id,
                project=project_id,
                record=expt_record_name,
                name=expt_name, 
                updates={'mlflow_uri': tracking_uri}
            )

        stripped_expt_mlflow_details = self.__rpc_formatter.strip_keys(
            record=expt_mlflow_record
        )
        return stripped_expt_mlflow_details


    def initialise_mlflow_run(
        self, 
        collab_id: str, 
        project_id: str, 
        expt_id: str, 
        run_id: str
    ) -> Dict[str, str]:
        """ Initialises a MLFLow run under a specified experiment of a project.
            Initial run hyperparameters will be logged, and MLFlow run id will
            be stored for subsequent analysis.

        Args:
            project_id (str): REST-RPC ID of specified project
            expt_id (str): REST-RPC ID of specified experiment
            run_id (str): REST-RPC ID of specified run
        """
        # Retrieve parent MLFlow experiment metadata (if available)
        expt_mlflow_record = self.retrieve_mlflow_experiment(
            collab_id=collab_id, 
            project_id=project_id, 
            expt_id=expt_id
        )
        expt_mlflow_id = expt_mlflow_record.get('mlflow_id')
        expt_mlflow_uri = expt_mlflow_record.get('mlflow_uri')

        if not expt_mlflow_id:
            logging.error(
                "MLFlow experiment has not been initialised!",
                ID_path=SOURCE_FILE,
                ID_class=MLFlogger.__name__,
                ID_function=MLFlogger.initialise_mlflow_run.__name__
            )
            raise RuntimeError("MLFlow experiment has not been initialised!")

        run_name = self._generate_run_name(collab_id, project_id, expt_id, run_id)
        run_record_name = self._generate_record_name(
            collab_id, project_id, expt_id, run_id,
            mlflow_type="run"
        )
        run_mlflow_record = self.mlf_records.read(
            collaboration=collab_id,
            project=project_id, 
            record=run_record_name,
            name=run_name
        )
        if not run_mlflow_record:

            with mlflow.start_run(
                experiment_id=expt_mlflow_id, 
                run_name=run_name
            ) as mlf_run:

                # Retrieve run details from database
                run_details = self.__run_records.read(
                    collab_id=collab_id, 
                    project_id=project_id, 
                    expt_id=expt_id, 
                    run_id=run_id
                )
                stripped_run_details = self.__rpc_formatter.strip_keys(
                    record=run_details,
                    concise=True
                )

                mlflow.log_params(stripped_run_details)

                # Save the MLFlow ID mapping
                run_mlflow_id = mlf_run.info.run_id
                run_mlflow_details = {
                    'collaboration': collab_id,
                    'project': project_id,
                    'record': run_record_name,
                    'name': run_name,
                    'mlflow_type': 'run',
                    'mlflow_id': run_mlflow_id,
                    'mlflow_uri': expt_mlflow_record['mlflow_uri'] # same as expt
                }
                run_mlflow_record = self.mlf_records.create(
                    collaboration=collab_id,
                    project=project_id,
                    record=run_record_name,
                    name=run_name, 
                    details=run_mlflow_details
                )

        if run_mlflow_record.get('mlflow_uri') != expt_mlflow_uri:
            run_mlflow_record = self.mlf_records.update(
                collaboration=collab_id,
                project=project_id,
                record=run_record_name,
                name=run_name, 
                updates={'mlflow_uri': expt_mlflow_record['mlflow_uri']}
            )

        stripped_run_mlflow_details = self.__rpc_formatter.strip_keys(
            record=run_mlflow_record
        )
        return stripped_run_mlflow_details       


    def log_losses(
        self, 
        collab_id: str, 
        project_id: str, 
        expt_id: str, 
        run_id: str
    ):
        """ Registers all cached losses, be it global or local, obtained from
            federated training into MLFlow.

        Args:
            project_id (str): REST-RPC ID of specified project
            expt_id (str): REST-RPC ID of specified experiment
            run_id (str): REST-RPC ID of specified run
        Returns:
            Stripped metadata (dict)
        """
        # Retrieve parent MLFlow experiment metadata (if available)
        expt_mlflow_record = self.retrieve_mlflow_experiment(
            collab_id=collab_id, 
            project_id=project_id, 
            expt_id=expt_id
        )
        expt_mlflow_id = expt_mlflow_record.get('mlflow_id')

        if not expt_mlflow_id:
            logging.error(
                "MLFlow experiment has not been initialised!",
                ID_path=SOURCE_FILE,
                ID_class=MLFlogger.__name__,
                ID_function=MLFlogger.log_losses.__name__
            )
            raise RuntimeError("MLFlow experiment has not been initialised!")

        # Search for run session to update entry, not create a new one
        run_mlflow_record = self.retrieve_mlflow_run(
            collab_id=collab_id, 
            project_id=project_id, 
            expt_id=expt_id, 
            run_id=run_id
        )
        run_mlflow_id = run_mlflow_record.get('mlflow_id')

        if not run_mlflow_id:
            logging.error(
                "MLFlow run has not been initialised!",
                ID_path=SOURCE_FILE,
                ID_class=MLFlogger.__name__,
                ID_function=MLFlogger.log_losses.__name__
            )
            raise RuntimeError("MLFlow run has not been initialised!")

        # Retrieve all model metadata from storage
        model_metadata = self.__model_records.read(
            collab_id=collab_id, 
            project_id=project_id, 
            expt_id=expt_id, 
            run_id=run_id
        )
        stripped_metadata = self.__rpc_formatter.strip_keys(
            record=model_metadata, 
            concise=True
        ) if model_metadata else model_metadata

        if stripped_metadata:

            with mlflow.start_run(
                experiment_id=expt_mlflow_id, 
                run_id=run_mlflow_id # continue existing run
            ) as mlf_run:

                # Extract loss histories
                for m_type, metadata in stripped_metadata.items():

                    loss_history_path = metadata['loss_history']
                    origin = metadata['origin']

                    with open(loss_history_path, 'r') as lh:
                        loss_history = json.load(lh)

                    if m_type == 'global':
                        for meta, losses in loss_history.items():
                            for round_idx, loss, in losses.items():
                                mlflow.log_metric(
                                    key=f"global_{meta}_loss", 
                                    value=loss, 
                                    step=int(round_idx)
                                )

                    else:
                        for round_idx, loss, in losses.items():
                            mlflow.log_metric(
                                key=f"{origin}_local_loss", 
                                value=loss, 
                                step=int(round_idx)
                            )

        return stripped_metadata

    
    def log_model_performance(
        self, 
        collab_id: str,
        project_id: str, 
        expt_id: str, 
        run_id: str,
        statistics: dict
    ) -> dict:
        """ Using all cached model checkpoints, log the performance statistics 
            of models, be it global or local, to MLFlow at round (for global) or
            epoch (for local) level.

        Args:
            project_id (str): REST-RPC ID of specified project
            expt_id (str): REST-RPC ID of specified experiment
            run_id (str): REST-RPC ID of specified run
            statistics (dict): Inference statistics polled from workers
        Returns:
            MLFLow run details (dict)
        """
        # Retrieve parent MLFlow experiment metadata (if available)
        expt_mlflow_record = self.retrieve_mlflow_experiment(
            collab_id=collab_id, 
            project_id=project_id, 
            expt_id=expt_id
        )
        expt_mlflow_id = expt_mlflow_record.get('mlflow_id')

        if not expt_mlflow_id:
            logging.error(
                "MLFlow experiment has not been initialised!",
                ID_path=SOURCE_FILE,
                ID_class=MLFlogger.__name__,
                ID_function=MLFlogger.log_losses.__name__
            )
            raise RuntimeError("MLFlow experiment has not been initialised!")

        # Search for run session to update entry, not create a new one
        run_mlflow_record = self.retrieve_mlflow_run(
            collab_id=collab_id, 
            project_id=project_id, 
            expt_id=expt_id, 
            run_id=run_id
        )
        run_mlflow_id = run_mlflow_record.get('mlflow_id')

        if not run_mlflow_id:
            logging.error(
                "MLFlow run has not been initialised!",
                ID_path=SOURCE_FILE,
                ID_class=MLFlogger.__name__,
                ID_function=MLFlogger.log_losses.__name__
            )
            raise RuntimeError("MLFlow run has not been initialised!")

        with mlflow.start_run(
            experiment_id=expt_mlflow_id, 
            run_id=run_mlflow_id
        ) as mlf_run:

            # Store output metadata into database
            for _, inference_stats in statistics.items():
                for _, meta_stats in inference_stats.items():

                    # Log statistics to MLFlow for analysis
                    stats = meta_stats.get('statistics', {})
                    for stat_type, stat_value in stats.items():

                        if isinstance(stat_value, list):
                            for val_idx, value in enumerate(stat_value):
                                mlflow.log_metric(
                                    key=f"{stat_type}_class_{val_idx}", 
                                    value=value, 
                                    step=int(val_idx+1)
                                )

                        else:
                            mlflow.log_metric(key=stat_type, value=stat_value)

        stripped_mlflow_run_details = self.__rpc_formatter.strip_keys(
            run_mlflow_record, 
            concise=True
        )
        return stripped_mlflow_run_details


    def log_artifacts(
        self, 
        collab_id: str,
        project_id: str, 
        expt_id: str, 
        run_id: str
    ) -> str:
        """ Log all artifacts produced (i.e. local & global models, losses etc.)
            during the specified federated run.

        Args:
            project_id (str): REST-RPC ID of specified project
            expt_id (str): REST-RPC ID of specified experiment
            run_id (str): REST-RPC ID of specified run
            statistics (dict): Inference statistics polled from workers
        Returns:
            MLFLow run details (dict)
        """
        # Retrieve parent MLFlow experiment metadata (if available)
        expt_mlflow_record = self.retrieve_mlflow_experiment(
            collab_id=collab_id, 
            project_id=project_id, 
            expt_id=expt_id
        )
        expt_mlflow_id = expt_mlflow_record.get('mlflow_id')

        if not expt_mlflow_id:
            logging.error(
                "MLFlow experiment has not been initialised!",
                ID_path=SOURCE_FILE,
                ID_class=MLFlogger.__name__,
                ID_function=MLFlogger.log_losses.__name__
            )
            raise RuntimeError("MLFlow experiment has not been initialised!")

        # Search for run session to update entry, not create a new one
        run_mlflow_record = self.retrieve_mlflow_run(
            collab_id=collab_id, 
            project_id=project_id, 
            expt_id=expt_id, 
            run_id=run_id
        )
        run_mlflow_id = run_mlflow_record.get('mlflow_id')

        if not run_mlflow_id:
            logging.error(
                "MLFlow run has not been initialised!",
                ID_path=SOURCE_FILE,
                ID_class=MLFlogger.__name__,
                ID_function=MLFlogger.log_losses.__name__
            )
            raise RuntimeError("MLFlow run has not been initialised!")
       
        with mlflow.start_run(
            experiment_id=expt_mlflow_id, 
            run_id=run_mlflow_id # continue existing run
        ) as mlf_run:

            results_dir = os.path.join(out_dir, collab_id, project_id, expt_id, run_id)           
            mlflow.log_artifacts(local_dir=results_dir)

        return results_dir

    ##################
    # Core Functions #
    ##################

    def log(self, accumulations: dict) -> List[str]:
        """ Wrapper function that processes statistics accumulated from 
            inference.

        Args:
            accumulations (dict): Accumulated statistics from inferring
                different project-expt-run combinations
        Returns:
            List of MLFlow run IDs from all runs executed (list(str))
        """

        ###########################
        # Implementation Footnote #
        ###########################

        # [Causes]
        # Depending on where your MLFlow logs are, the accessibility of the
        # resource changes.

        # [Problems]
        # As Synergos MLOps is an optional component, the orchestrator may or
        # may not install it. Also, depending on the location of deployment
        # (i.e. local filesystem vs remote tracking server), connection URI
        # may vary.

        # [Solution]
        # By default, ALWAYS generate local MLFlow caches. If a remote tracking
        # URI was specified, then attempt to send logs over. That way, clients
        # will always have the option of choosing which way they would to serve
        # their MLOps information

        has_remote_tracking = self.remote_uri is not None
        tracking_uris = (
            [mlflow_dir, self.remote_uri] 
            if has_remote_tracking 
            else [mlflow_dir]
        ) 

        jobs_ran = []
        for tracking_uri in tracking_uris:
            for combination_key, statistics in accumulations.items():

                collab_id, project_id, expt_id, run_id = combination_key

                self.initialise_mlflow_experiment(
                    collab_id=collab_id,
                    project_id=project_id,
                    expt_id=expt_id,
                    tracking_uri=tracking_uri
                )
                run_mlflow_details = self.initialise_mlflow_run(
                    collab_id=collab_id,
                    project_id=project_id,
                    expt_id=expt_id,
                    run_id=run_id
                )
                self.log_losses(
                    collab_id=collab_id,
                    project_id=project_id,
                    expt_id=expt_id,
                    run_id=run_id
                )
                self.log_model_performance(
                    collab_id=collab_id,
                    project_id=project_id,
                    expt_id=expt_id,
                    run_id=run_id,
                    statistics=statistics
                )
                self.log_artifacts(
                    collab_id=collab_id,
                    project_id=project_id,
                    expt_id=expt_id,
                    run_id=run_id
                )

                if run_mlflow_details['mlflow_id'] not in jobs_ran:
                    jobs_ran.append(run_mlflow_details['mlflow_id'])
        
        return jobs_ran