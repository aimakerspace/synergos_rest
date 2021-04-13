#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import copy
import os
import random
import uuid
from datetime import datetime, timedelta
from string import Template
from typing import Dict, List, Tuple, Union

# Libs
import pytest
import tinydb

# Custom
from synarchive.base import Records, TopicalRecords, AssociationRecords
from synarchive.connection import (
    CollaborationRecords, ProjectRecords, ExperimentRecords, RunRecords,
    ParticipantRecords, RegistrationRecords, TagRecords
)
from synarchive.training import AlignmentRecords, ModelRecords
from synarchive.evaluation import MLFRecords, ValidationRecords, PredictionRecords

##################
# Configurations #
##################

TEST_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "test_database.json"
)

TTP_SUBJECTS = [
    "Collaboration", 
    "Project", "Experiment", "Run", "Model", "Validation", "Prediction",
    "MLFlow",
    "Registration", "Tag", "Alignment" 
]

WORKER_SUBJECTS = [
    "Participant",
    "Registration", "Tag", "Alignment"
]

ID_KEYS = {
    'collaboration': 'collab_id',
    'project': 'project_id',
    'experiment': 'expt_id',
    'run': 'run_id',
    'participant': 'participant_id',
    'registration': 'registration_id',
    'tag': 'tag_id',
    'alignment': 'alignment_id',
    'model': 'model_id',
    'validation': 'val_id',
    'prediction': 'pred_id',
    'mlflow': 'name'
}

RELATIONS_MAPPINGS = {
    # TTP-orchestrated mappings
    'collaboration': [
        "Project", "Experiment", "Run", "Model", "Validation", "Prediction",
        "Registration", "Tag", "Alignment" 
    ],
    'project': [
        "Experiment", "Run", "Model", "Validation", "Prediction",
        "Registration", "Tag", "Alignment" 
    ],
    'experiment': ["Run", "Model", "Validation", "Prediction"],
    'run': ["Model", "Validation", "Prediction"],
    'model': ["Validation", "Prediction"],
    'validation': [],
    'prediction': [],
    
    # Worker-orchestrated mappings
    'participant': [
        "Registration", "Tag", "Alignment", 
        "Validation", "Prediction"
    ],
    'registration': ["Tag", "Alignment"],
    'tag': ["Alignment"],
    'alignment': [],

    # Misc mappings
    'mlflow': []
}

KEY_ID_MAPPINGS = {
    # TTP-orchestrated mappings
    'collaboration': [
        ID_KEYS[r_type] 
        for r_type in ["collaboration"]
    ],
    'project': [
        ID_KEYS[r_type] 
        for r_type in ["collaboration", "project"]
    ],
    'experiment': [
        ID_KEYS[r_type] 
        for r_type in ["collaboration", "project", "experiment"]
    ],
    'run': [
        ID_KEYS[r_type] 
        for r_type in ["collaboration", "project", "experiment", "run"]
    ],
    'model': [
        ID_KEYS[r_type] 
        for r_type in ["collaboration", "project", "experiment", "run"]
    ],
    'validation': [
        ID_KEYS[r_type] 
        for r_type in ["participant", "collaboration", "project", "experiment", "run"]
    ],
    'prediction': [
        ID_KEYS[r_type] 
        for r_type in ["participant", "collaboration", "project", "experiment", "run"]
    ],
    
    # Worker-orchestrated mappings
    'participant': [
        ID_KEYS[r_type] 
        for r_type in ["participant"]
    ],
    'registration': [
        ID_KEYS[r_type] 
        for r_type in ["participant", "collaboration", "project"]
    ],
    'tag': [
        ID_KEYS[r_type] 
        for r_type in ["participant", "collaboration", "project"]
    ],
    'alignment':[
        ID_KEYS[r_type] 
        for r_type in ["participant", "collaboration", "project"]
    ],

    # Misc mappings
    'mlflow':["collaboration", "project", "name"],
}

LINK_MAPPINGS = {
    # TTP-orchestrated mappings
    'model': [],
    'validation': [],   # no need to delete predictions if deleted
    'prediction': [],   # no need to delete validations if deleted
    
    # Worker-orchestrated mappings
    'registration': [],
    'tag': ["Registration"],
    'alignment': ["Registration", "Tag"]
}

LINK_ID_MAPPINGS = {
    r_type: (
        [ID_KEYS[r_type]] +
        [ID_KEYS[subject.lower()] for subject in links]
    )
    for r_type, links in LINK_MAPPINGS.items()  
}

COLLAB_PREFIX = Template("COLLAB_$collab_id")
PROJECT_PREFIX = Template("PROJECT_$project_id")
EXPT_PREFIX = Template("EXPT_$expt_id")
RUN_PREFIX = Template("RUN_$run_id")
PARTICIPANT_PREFIX = Template("PARTICIPANT_$participant_id")
PREFIXES = {
    'collaboration': COLLAB_PREFIX,
    'project': PROJECT_PREFIX,
    'experiment': EXPT_PREFIX,
    'run': RUN_PREFIX,
    'participant': PARTICIPANT_PREFIX
}

###########
# Helpers #
###########

def simulate_choice() -> bool:
    """ Simulates a boolean choice.

    Returns:
        A random choice (bool)
    """
    return random.random() < 0.5


def simulate_ip() -> str:
    """ Simulates a random IP address

    Returns:
        A random IP address (str)
    """
    return "{}.{}.{}.{}".format(
        random.randint(1, 1000),
        random.randint(1, 1000),
        random.randint(1, 1000),
        random.randint(1, 1000)
    )


def simulate_port() -> int:
    """ Simulates a random port

    Returns:
        A random port (int)
    """
    return random.randint(1001, 65535)


def simulate_setup(
    collabs: int = 2,
    projects: int = 2,
    experiments: int = 2,
    runs: int = 3,
    participants: int = 5
) -> Dict[str, List[Dict[str, str]]]:
    """ Automates generation of test keys for core topical records.

    Args:
        collabs (int): No. of collaborations to simulate
        projects (int): No. of projects per collab to simulate
        experiments (int): No. of experiments per project to simulate
        runs (int): No. of runs per experiment to simulate
        participants (int): No. of participants per collaboration to simulate
    Returns:
        All simulated keys (dict)
    """
    def construct_id(core_idx: int):
        return "-".join([str(core_idx), str(uuid.uuid4())])

    def update_topic_keys(key_dict, topic, new_key):
        topic_entries = key_dict.get(topic, [])
        topic_entries.append(new_key)
        key_dict.update({topic: topic_entries})

    def create_key(key_dict, topic, idx, hierarchy):
        topic_prefix = PREFIXES.get(topic)
        topic_id_key = ID_KEYS.get(topic)
        topic_id = topic_prefix.substitute({topic_id_key: construct_id(idx)})
        topic_key = {**hierarchy, topic_id_key: topic_id}
        update_topic_keys(key_dict, topic, topic_key)
        return topic_key

    # Generate unique collaborations
    all_keys = {}
    for collab_idx in range(collabs): 
        collab_key = create_key(
            key_dict=all_keys, 
            topic="collaboration", 
            idx=collab_idx, 
            hierarchy={}
        )

        # Generate unique projects
        for project_idx in range(projects):
            project_key = create_key(
                key_dict=all_keys, 
                topic="project", 
                idx=project_idx, 
                hierarchy=collab_key
            )

            # Generate unique experiments
            for expt_idx in range(experiments):
                expt_key = create_key(
                    key_dict=all_keys, 
                    topic="experiment", 
                    idx=expt_idx, 
                    hierarchy=project_key
                )

                # Generate unique runs
                for run_idx in range(runs):
                    run_key = create_key(
                        key_dict=all_keys, 
                        topic="run", 
                        idx=run_idx, 
                        hierarchy=expt_key
                    )

    # Generate unique participants
    for participant_idx in range(participants):
        participant_key = create_key(
            key_dict=all_keys, 
            topic="participant", 
            idx=participant_idx, 
            hierarchy={}
        )

    return all_keys


def generate_record_info() -> Tuple[
    str,
    Dict[str, str],
    Dict[str, Union[int, float, str, list, dict]]
]:
    """ Generates unstructure information for testing general database 
        operations.

    Args:
        include_key (bool): Toggles if generated info should include a key
    Returns:
        Test key (str)
        Test Composite ID (dict)
        Test information (dict)
    """
    test_key = f"KEY-{uuid.uuid4()}"
    test_ids = {
        f'TEST_ID{i}-{uuid.uuid4()}': f"ID{i}-{uuid.uuid4()}"
        for i in range(random.randint(1, 5))
    }
    test_info = {
        'int_field': random.randint(-1000, 1000),
        'float_field': random.random(),
        'str_field': str(uuid.uuid4()),
        'list_field': [i for i in range(random.randint(1, 10))],
        'dict_field': {f'key_{i}':i for i in range(random.randint(1, 10))}
    }

    return test_key, test_ids, test_info


def generate_federated_combination() -> Tuple[str]:
    """ Generates a federated combination of composite IDs that uniquely 
        documents essential components of a single federated cycle

    Returns:
        Collaboration ID (str)
        Project ID       (str)
        Experiment ID    (str)
        Run ID           (str)
        Participant ID   (str)
    """
    test_keys = simulate_setup()
    
    federated_combination = random.choice(test_keys['run']) # lowest hierarchy
    collab_id = federated_combination[ID_KEYS['collaboration']]
    project_id = federated_combination[ID_KEYS['project']]
    expt_id = federated_combination[ID_KEYS['experiment']]
    run_id = federated_combination[ID_KEYS['run']]
    
    participant = random.choice(test_keys['participant'])
    participant_id = participant.get(ID_KEYS['participant'])
    return collab_id, project_id, expt_id, run_id, participant_id


def generate_collaboration_info() -> dict:
    """ Generates metadata used for creating a collaboration record

    Returns:
        Collaboration metadata (dict) 
    """
    return {
        # Catalogue Connection
        'catalogue_host': simulate_ip(),
        'catalogue_port': simulate_port(),
        # Logger Connection
        'logger_host': simulate_ip(),
        'logger_ports': {
            'sysmetrics': simulate_port(),
            'director': simulate_port(),
            'ttp': simulate_port(),
            'worker': simulate_port(),
        },
        # Meter Connection
        'meter_host': simulate_ip(),
        'meter_port': simulate_port(),
        # MLOps Connection
        'mlops_host': simulate_ip(),
        'mlops_port': simulate_port(),
        # MQ Connection
        'mq_host': simulate_ip(),
        'mq_port': simulate_port(),
        # UI Connection
        'ui_host': simulate_ip(),
        'ui_port': simulate_port()
    }


def generate_project_info() -> dict:
    """ Generates metadata used for creating a project record

    Returns:
        Project metadata (dict) 
    """
    return {
        "universe_alignment": [],
        "incentives": {},
        "start_at": datetime.strptime(
            datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            '%Y-%m-%d %H:%M:%S'
        ) + timedelta(hours=random.randint(0, 1000))
    }


def generate_experiment_info() -> dict:
    """ Generates metadata used for creating an experiment record

    Returns:
        Experiment metadata (dict) 
    """
    return {
        "model": [
            {
                "is_input": True,
                "structure": {
                    "in_features": random.randint(1, 1000),
                    "out_features": random.randint(1, 100),
                    "bias": simulate_choice()
                },
                "l_type": "linear",
                "activation": "sigmoid"
            }
        ]
    }


def generate_run_info() -> Dict[str, Union[int, float, bool, str]]:
    """ Generates metadata used for creating a run record

    Returns:
        Run metadata (dict) 
    """
    return {
        "input_size": random.randint(1, 1000),
        "output_size": random.randint(1, 100),
        "batch_size": random.randint(1, 5000),
        "lr": random.random(),
        "weight_decay": random.random(),
        "rounds": random.randint(0, 100),
        "epochs": random.randint(0, 100),
        "lr_decay": random.random(),
        "seed": random.randint(1, 1000),
        "is_condensed": simulate_choice(),
        "precision_fractional": random.randint(1, 10),
        "use_CLR": simulate_choice(),
        "mu": random.random(),
        "reduction": random.choice(["none", "mean", "sum"]),
        "l1_lambda": random.random(),
        "l2_lambda": random.random(),
        "dampening": random.random(),
        "base_lr": random.random(),
        "max_lr": random.random(),
        "step_size_up": random.randint(1, 10),
        "step_size_down": random.randint(1, 10),
        "mode": random.choice(["triangular", "triangular2", "exp_range"]),
        "gamma": random.random(),
        "scale_mode": random.choice(["cycle", "iterations"]),
        "cycle_momentum": simulate_choice(),
        "base_momentum": random.random(),
        "max_momentum": random.random(),
        "last_epoch": random.randint(1, 10),
        "patience": random.randint(1, 10),
        "delta": random.random(),
        "cumulative_delta": simulate_choice()
    }


def generate_participant_info() -> Dict[str, Union[str, int, bool]]:
    """ Generates metadata used for creating a participant record

    Returns:
        Participant metadata (dict) 
    """
    return {}


def generate_registration_info() -> dict:
    """ Generates metadata used for creating a registration record

    Returns:
        Registration metadata (dict) 
    """
    grid_count = random.randint(1, 10)
    channels = {
        f"node_{grid_idx}": {
            "host": simulate_ip(),
            "f_port": simulate_port(),
            "port": simulate_port(),
            "log_msgs": simulate_choice(),
            "verbose": simulate_choice()
        } 
        for grid_idx in range(grid_count)
    }
    return {'role': "guest", 'n_count': grid_count, **channels}


def generate_tag_info() -> dict:
    """ Generates metadata used for creating a tag record

    Returns:
        Tag metadata (dict) 
    """
    def simulate_tag():
        dataset_name = "test_dataset"
        dataset_set = f"set_{str(random.randint(0, 10))}"
        version = f"version_{str(random.randint(0, 10))}"
        return [dataset_name, dataset_set, version]

    return {
        meta: [simulate_tag() for _ in range(random.randint(1, 10))] 
        for meta in ["train", "evaluate", "predict"]
    }


def generate_alignment_info() -> dict:
    """ Generates metadata used for creating an alignment record

    Returns:
        Alignment metadata (dict) 
    """
    def simulate_alignment() -> List[int]:
        X_alignments = sorted([
            random.randint(0, 1000) 
            for _ in range(random.randint(1, 100))
        ])
        y_alignments = sorted([
            random.randint(0, 10) 
            for _ in range(random.randint(1, 10))
        ])
        return {'X': X_alignments, 'y': y_alignments}

    return {
        meta: simulate_alignment() 
        for meta in ["train", "evaluate", "predict"]
    }


def generate_model_info() -> dict:
    """ Generates metadata used for creating a model record

    Returns:
        Model metadata (dict) 
    """
    return {
        "global": {
            "checkpoints": {
                "round_0": {
                    "epoch_0": {
                        "loss_history": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_0/epoch_0/global_loss_history.json",
                        "path": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_0/epoch_0/global_model.pt"
                    },
                    "epoch_1": {
                        "loss_history": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_0/epoch_1/global_loss_history.json",
                        "path": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_0/epoch_1/global_model.pt"
                    }
                },
                "round_1": {
                    "epoch_0": {
                        "loss_history": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_1/epoch_0/global_loss_history.json",
                        "path": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_1/epoch_0/global_model.pt"
                    },
                    "epoch_1": {
                        "loss_history": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_1/epoch_1/global_loss_history.json",
                        "path": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_1/epoch_1/global_model.pt"
                    }
                },
                "round_2": {
                    "epoch_0": {
                        "loss_history": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_2/epoch_0/global_loss_history.json",
                        "path": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_2/epoch_0/global_model.pt"
                    },
                    "epoch_1": {
                        "loss_history": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_2/epoch_1/global_loss_history.json",
                        "path": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_2/epoch_1/global_model.pt"
                    }
                },
                "round_3": {
                    "epoch_0": {
                        "loss_history": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_3/epoch_0/global_loss_history.json",
                        "path": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_3/epoch_0/global_model.pt"
                    },
                    "epoch_1": {
                        "loss_history": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_3/epoch_1/global_loss_history.json",
                        "path": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_3/epoch_1/global_model.pt"
                    }
                },
                "round_4": {
                    "epoch_0": {
                        "loss_history": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_4/epoch_0/global_loss_history.json",
                        "path": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_4/epoch_0/global_model.pt"
                    },
                    "epoch_1": {
                        "loss_history": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_4/epoch_1/global_loss_history.json",
                        "path": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_4/epoch_1/global_model.pt"
                    }
                }
            },
            "loss_history": "/ttp/outputs/test_project_1/test_experiment/test_run/global_loss_history.json",
            "origin": "ttp",
            "path": "/ttp/outputs/test_project_1/test_experiment/test_run/global_model.pt"
        },
        "local_1": {
            "checkpoints": {
                "round_0": {
                    "epoch_0": {
                        "loss_history": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_0/epoch_0/local_loss_history_test_participant_1.json",
                        "path": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_0/epoch_0/local_model_test_participant_1.pt"
                    },
                    "epoch_1": {
                        "loss_history": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_0/epoch_1/local_loss_history_test_participant_1.json",
                        "path": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_0/epoch_1/local_model_test_participant_1.pt"
                    }
                },
                "round_1": {
                    "epoch_0": {
                        "loss_history": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_1/epoch_0/local_loss_history_test_participant_1.json",
                        "path": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_1/epoch_0/local_model_test_participant_1.pt"
                    },
                    "epoch_1": {
                        "loss_history": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_1/epoch_1/local_loss_history_test_participant_1.json",
                        "path": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_1/epoch_1/local_model_test_participant_1.pt"
                    }
                },
                "round_2": {
                    "epoch_0": {
                        "loss_history": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_2/epoch_0/local_loss_history_test_participant_1.json",
                        "path": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_2/epoch_0/local_model_test_participant_1.pt"
                    },
                    "epoch_1": {
                        "loss_history": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_2/epoch_1/local_loss_history_test_participant_1.json",
                        "path": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_2/epoch_1/local_model_test_participant_1.pt"
                    }
                },
                "round_3": {
                    "epoch_0": {
                        "loss_history": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_3/epoch_0/local_loss_history_test_participant_1.json",
                        "path": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_3/epoch_0/local_model_test_participant_1.pt"
                    },
                    "epoch_1": {
                        "loss_history": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_3/epoch_1/local_loss_history_test_participant_1.json",
                        "path": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_3/epoch_1/local_model_test_participant_1.pt"
                    }
                },
                "round_4": {
                    "epoch_0": {
                        "loss_history": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_4/epoch_0/local_loss_history_test_participant_1.json",
                        "path": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_4/epoch_0/local_model_test_participant_1.pt"
                    },
                    "epoch_1": {
                        "loss_history": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_4/epoch_1/local_loss_history_test_participant_1.json",
                        "path": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_4/epoch_1/local_model_test_participant_1.pt"
                    }
                }
            },
            "loss_history": "/ttp/outputs/test_project_1/test_experiment/test_run/local_loss_history_test_participant_1.json",
            "origin": "test_participant_1",
            "path": "/ttp/outputs/test_project_1/test_experiment/test_run/local_model_test_participant_1.pt"
        },
        "local_2": {
            "checkpoints": {
                "round_0": {
                    "epoch_0": {
                        "loss_history": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_0/epoch_0/local_loss_history_test_participant_2.json",
                        "path": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_0/epoch_0/local_model_test_participant_2.pt"
                    },
                    "epoch_1": {
                        "loss_history": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_0/epoch_1/local_loss_history_test_participant_2.json",
                        "path": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_0/epoch_1/local_model_test_participant_2.pt"
                    }
                },
                "round_1": {
                    "epoch_0": {
                        "loss_history": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_1/epoch_0/local_loss_history_test_participant_2.json",
                        "path": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_1/epoch_0/local_model_test_participant_2.pt"
                    },
                    "epoch_1": {
                        "loss_history": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_1/epoch_1/local_loss_history_test_participant_2.json",
                        "path": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_1/epoch_1/local_model_test_participant_2.pt"
                    }
                },
                "round_2": {
                    "epoch_0": {
                        "loss_history": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_2/epoch_0/local_loss_history_test_participant_2.json",
                        "path": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_2/epoch_0/local_model_test_participant_2.pt"
                    },
                    "epoch_1": {
                        "loss_history": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_2/epoch_1/local_loss_history_test_participant_2.json",
                        "path": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_2/epoch_1/local_model_test_participant_2.pt"
                    }
                },
                "round_3": {
                    "epoch_0": {
                        "loss_history": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_3/epoch_0/local_loss_history_test_participant_2.json",
                        "path": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_3/epoch_0/local_model_test_participant_2.pt"
                    },
                    "epoch_1": {
                        "loss_history": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_3/epoch_1/local_loss_history_test_participant_2.json",
                        "path": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_3/epoch_1/local_model_test_participant_2.pt"
                    }
                },
                "round_4": {
                    "epoch_0": {
                        "loss_history": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_4/epoch_0/local_loss_history_test_participant_2.json",
                        "path": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_4/epoch_0/local_model_test_participant_2.pt"
                    },
                    "epoch_1": {
                        "loss_history": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_4/epoch_1/local_loss_history_test_participant_2.json",
                        "path": "/ttp/outputs/test_project_1/test_experiment/test_run/checkpoints/round_4/epoch_1/local_model_test_participant_2.pt"
                    }
                }
            },
            "loss_history": "/ttp/outputs/test_project_1/test_experiment/test_run/local_loss_history_test_participant_2.json",
            "origin": "test_participant_2",
            "path": "/ttp/outputs/test_project_1/test_experiment/test_run/local_model_test_participant_2.pt"
        }
    }


def generate_mlflow_info() -> Tuple[Dict[str, str]]:
    """ Generates metadata used for creating an MLFLow record

    Returns:
        MLFlow experiment metadata (dict) 
        MLFlow run metadata        (dict)
    """
    expt_tracked = {
        "mlflow_id": "0",
        "mlflow_type": "experiment",
        "mlflow_uri": "/ttp/mlflow/test_collab/test_project_1",
        "name": "test_experiment",
        "project": "test_project_1",
        "collaboration": "test_collab"
    }
    run_tracked = {
        "mlflow_id": "f81c4ff2c4704a0da07da1d16b4dfe9e",
        "mlflow_type": "run",
        "mlflow_uri": "/ttp/mlflow/test_collab/test_project_1",
        "name": "test_run",
        "project": "test_project_1",
        "collaboration": "test_collab"
    }
    return expt_tracked, run_tracked


def generate_inference_info(
    action: str, 
    label_count: int = 1, 
    meta: str = "evaluate"
) -> dict:
    """ Generates metadata used for creating an inference record

    Returns:
        Inference metadata (dict) 
    """
    def simulate_float_stats():
        return [random.random() for _ in range(label_count)]

    def simulate_int_stats():
        return [random.randint(0, 5000) for _ in range(label_count)]

    classification_stats = {
        "FDRs": simulate_float_stats(),
        "FNRs": simulate_float_stats(),
        "FNs": simulate_int_stats(),
        "FPRs": simulate_float_stats(),
        "FPs": simulate_int_stats(),
        "NPVs": simulate_float_stats(),
        "PPVs": simulate_float_stats(),
        "TNRs": simulate_float_stats(),
        "TNs": simulate_int_stats(),
        "TPRs": simulate_float_stats(),
        "TPs": simulate_int_stats(),
        "accuracy": simulate_float_stats(),
        "f_score": simulate_float_stats(),
        "pr_auc_score": simulate_float_stats(),
        "roc_auc_score": simulate_float_stats()
    }

    regression_stats = {
        'R2': random.random(),
        'MSE': random.random(), 
        'MAE': random.random()
    }

    return {
        meta: {
            "res_path": f"/worker/outputs/test_collaboration/test_project/test_experiment/test_run/{meta}/inference_statistics_{meta}.json",
            "statistics": (
                regression_stats 
                if action == "regress" 
                else classification_stats
            )
        }
    }


def reset_database(archive: Records) -> None:
    """ Erases all existing data stored in an archive's database

    Args:
        archive (Records): Specified archive to be resetted
    """
    database = archive.load_database()
    database.purge()


########################
# Evaluative Functions #
########################

def check_field_equivalence(
    records: Records, 
    subject: str, 
    key: str, 
    new_record: Dict[str, Union[int, float, str, list, dict]]
) -> None:
    """ Tests if the fundamental insertion/update of record entries is correct

    # C1: Check that custom 'created_at' field was applied
    # C2: Check that all record keys are the same
    # C3: Check that all record values are the same

    Args:
        subject (str): Table to be operated on
        new_record (dict): Information for creating a new record
        key (str): Primary key of the current table
    """
    database = records.load_database()
    subject_table = database.table(subject)
    # C1
    retrieved_record = subject_table.get(tinydb.where(key) == new_record[key])
    assert 'created_at' in retrieved_record.keys()
    retrieved_record.pop('created_at')
    for k,v in retrieved_record.items():
        # C2
        assert k in new_record.keys()
        # C3
        assert v == new_record[k]


def check_insertion_or_update(
    records: Records, 
    subject: str, 
    key: str, 
    new_record: Dict[str, Union[int, float, str, list, dict]]
) -> None:
    """ Tests if the fundamental insertion/update of record entries is in-place

    # C1: Check that specified record for subject given key exists
    # C2: Check that if record is duplicated under the same key, total no. of
        records within the database remains the same
    # C3: Check that if record is duplicated under the same key, the original
        record is overwitten instead

    Args:

        subject (str): 
        key (dict(str, str)): Unique composite key that identifies a record
        new_record (tinydb.database.Document): 
    """
    database = records.load_database()
    subject_table = database.table(subject)
    # C1
    retrieved_record = subject_table.get(tinydb.where(key) == new_record[key])
    assert retrieved_record is not None
    # C2
    subject_record_count = len(subject_table.all())
    duplicated_record = records.create(subject, key, new_record)
    new_subject_record_count = len(subject_table.all())
    assert new_subject_record_count == subject_record_count
    # C3
    assert retrieved_record.doc_id == duplicated_record.doc_id


def check_key_equivalence(
    record: tinydb.database.Document, 
    r_type: str,
    ids: List[str]
) -> None:
    """ Tests if specified record is dynamic while being uniquely identifiable

    # C1: Check that specified record was dynamically created
    # C2: Check that specified record has a composite key
    # C3: Check that specified record was archived with correct substituent keys
    # C4: Check that specified record was archived with correct substituent IDs

    Args:
        record (tinydb.database.Document):
        ids (list(str)): List of IDs that make up record's composite key 
    """
    # Ensure that a cloned record is no different from its original
    cloned_record = copy.deepcopy(record)
    assert cloned_record == record
    # C1
    assert 'created_at' in cloned_record.keys()
    # C2
    assert "key" in cloned_record.keys()
    # C3
    key = cloned_record.pop('key')
    assert (set(key.keys()) == set(KEY_ID_MAPPINGS[r_type]))
    # C4
    assert set(ids) == set(key.values())


def check_relation_equivalence(
    record: tinydb.database.Document,
    r_type: str
) -> None:
    """ Tests if specified record is self-consistent and enforces hierarchy 
        downstream. The "relations" field only exists when a record is obtained
        through a query (i.e. .read(...), .read_all(...))

    # C1: Check hierarchy-enforcing field "relations" exist
    # C2: Check that all downstream relations have been captured 
    """
    # Ensure that a cloned record is no different from its original
    cloned_record = copy.deepcopy(record)
    assert cloned_record == record
    # C1
    assert 'relations' in cloned_record.keys()
    # C2
    relations = cloned_record.pop('relations')
    assert (set(relations.keys()) == set(RELATIONS_MAPPINGS[r_type]))


def check_link_equivalence(
    record: tinydb.database.Document,
    r_type: str    
) -> None:
    """ Tests if specified record is cross-consistent and enforces hierarchy
        upstream. (Links are dynamically generated composite keys that allow
        for upstream tracing, enforcing a different hierarchy from that 
        enforced by "key") 

    # C1: Check that composite key "link" exist for upstream transversal
    # C2: Check that the appropriate linked IDs are specified
    # C3: Check that keys in "link" are disjointed sets w.r.t "key" 
    """
    # Ensure that a cloned record is no different from its original
    cloned_record = copy.deepcopy(record)
    assert cloned_record == record
    # C1
    assert "link" in record.keys()
    # C2
    key = cloned_record.pop('key')
    link = cloned_record.pop('link')
    assert (set(link.keys()) == set(LINK_ID_MAPPINGS[r_type]))
    # C3
    assert not set(link.items()).issubset(set(key.items()))


def check_detail_equivalence(
    record: tinydb.database.Document, 
    details: dict
) -> None:
    """ Tests if a specified record contains the required details

    # C1: Check that specified record captured the correct specified details
    """
    # Ensure that a cloned record is no different from its original
    cloned_record = copy.deepcopy(record)
    assert cloned_record == record
    # C1
    cloned_record.pop('created_at')
    cloned_record.pop('key')
    cloned_record.pop('link', None)       # Only AssociationRecords has this!
    cloned_record.pop('relations', None)  # Only .read & .read_all has this!
    assert details == cloned_record


##########################
# Miscellaneous Fixtures #
##########################


######################
# Component Fixtures #
######################

@pytest.fixture(scope='session')
def record_env():
    records = Records(db_path=TEST_PATH)
    reset_database(records)

    test_subject = "RecordTest"
    test_key, test_ids, test_details = generate_record_info()
    _, _, test_updates = generate_record_info()

    return (
        records, 
        test_subject, test_key, test_ids, 
        test_details, test_updates
    )


@pytest.fixture(scope='session')
def topicalRecord_env():
    test_subject = "TopicalTest"
    test_relations = [f"RELATION-{i}" for i in range(random.randint(1, 10))]
    _, test_ids, test_details = generate_record_info()
    _, _, test_updates = generate_record_info()
    test_key = "key"
    test_identifier = random.choice(list(test_ids.keys()))

    topical_records = TopicalRecords(
        test_subject,
        test_identifier,
        TEST_PATH,
        *test_relations
    )
    reset_database(topical_records)

    # Set up related records
    records = Records(db_path=TEST_PATH)
    related_entries = {}
    for relation in test_relations:
        _, related_ids, related_details = generate_record_info()
        related_ids = {**related_ids, **test_ids} # overwrite with test IDs
        records.create(
            subject=relation,
            key=test_key,
            new_record={test_key: related_ids, **related_details}
        )
        related_entries[relation] = related_ids

    return (
        topical_records, 
        test_subject, test_key, test_ids, 
        test_details, test_updates,
        records, related_entries
    )


@pytest.fixture(scope='session')
def associationRecord_env():
    _, test_ids, _ = generate_record_info()
    test_key = "key"
    test_link = "link"

    # Generate Downstream & Upstream hierarchy
    SUBJECT_PREFIX = "AssociationTest"
    hierarchy = [f"{SUBJECT_PREFIX}-{i}" for i in range(random.randint(1, 10))]
    associated_record_hierarchy = []
    for idx in range(len(hierarchy)):
        associated_subject = hierarchy[idx]
        associated_identifier = f"assoc_id_{idx}"
        downstream_hierarchy = hierarchy[idx+1:]
        upstream_hierarchy = hierarchy[:idx]
        associated_records = AssociationRecords(
            associated_subject,
            associated_identifier,
            TEST_PATH,
            downstream_hierarchy,
            *upstream_hierarchy # load accumulated hierarchy
        )

        _, _, associated_details = generate_record_info()
        _, _, associated_updates = generate_record_info()

        # Store associated records in hierarchy
        level = (associated_records, associated_details, associated_updates)
        associated_record_hierarchy.append(level)

    records = Records(db_path=TEST_PATH)

    return (
        test_key, test_link, test_ids, 
        associated_record_hierarchy,
        records
    )


@pytest.fixture(scope='session')
def collab_env():
    collab_records = CollaborationRecords(db_path=TEST_PATH)
    reset_database(collab_records)

    # Simulate data
    collab_details = generate_collaboration_info() 
    collab_updates = generate_collaboration_info()

    (collab_id, project_id, expt_id, run_id, participant_id
    ) = generate_federated_combination()

    action = random.choice(["classify", "regress"])

    # Generate downstream hierarchy
    project_records = ProjectRecords(db_path=TEST_PATH)
    project_records.create(
        collab_id=collab_id,
        project_id=project_id,
        details=generate_project_info()
    )
    expt_records = ExperimentRecords(db_path=TEST_PATH)
    expt_records.create(
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id, 
        details=generate_experiment_info()
    )
    run_records = RunRecords(db_path=TEST_PATH)
    run_records.create(
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id, 
        run_id=run_id,
        details=generate_run_info()
    )
    model_records = ModelRecords(db_path=TEST_PATH)
    model_records.create( 
        collab_id=collab_id, 
        project_id=project_id, 
        expt_id=expt_id, 
        run_id=run_id,
        details=generate_model_info()
    )
    val_records = ValidationRecords(db_path=TEST_PATH)
    val_records.create(
        participant_id=participant_id, 
        collab_id=collab_id, 
        project_id=project_id, 
        expt_id=expt_id, 
        run_id=run_id,
        details=generate_inference_info(action, 10, "evaluate")
    )
    pred_records = PredictionRecords(db_path=TEST_PATH)
    pred_records.create(
        participant_id=participant_id, 
        collab_id=collab_id, 
        project_id=project_id, 
        expt_id=expt_id, 
        run_id=run_id,
        details=generate_inference_info(action, 10, "predict")
    )

    # Generate upstream hierarchy
    registration_records = RegistrationRecords(db_path=TEST_PATH)
    registration_records.create(
        collab_id=collab_id,
        project_id=project_id,
        participant_id=participant_id,
        details=generate_registration_info()
    )
    tag_records = TagRecords(db_path=TEST_PATH)
    tag_records.create( 
        collab_id=collab_id, 
        project_id=project_id,
        participant_id=participant_id, 
        details=generate_tag_info()
    )
    alignment_records = AlignmentRecords(db_path=TEST_PATH)
    alignment_records.create( 
        collab_id=collab_id, 
        project_id=project_id,
        participant_id=participant_id, 
        details=generate_alignment_info()
    )

    return (
        collab_records, 
        collab_details,
        collab_updates,
        (collab_id, project_id, expt_id, run_id, participant_id),
        (project_records, expt_records, run_records, model_records, 
         val_records, pred_records,
         registration_records, tag_records, alignment_records)
    )


@pytest.fixture(scope='session')
def project_env():
    project_records = ProjectRecords(db_path=TEST_PATH)
    reset_database(project_records)

    # Simulate data
    project_details = generate_project_info() 
    project_updates = generate_project_info()

    (collab_id, project_id, expt_id, run_id, participant_id
    ) = generate_federated_combination()

    action = random.choice(["classify", "regress"])

    # Generate downstream hierarchy
    expt_records = ExperimentRecords(db_path=TEST_PATH)
    expt_records.create(
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id, 
        details=generate_experiment_info()
    )
    run_records = RunRecords(db_path=TEST_PATH)
    run_records.create(
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id, 
        run_id=run_id,
        details=generate_run_info()
    )
    model_records = ModelRecords(db_path=TEST_PATH)
    model_records.create( 
        collab_id=collab_id, 
        project_id=project_id, 
        expt_id=expt_id, 
        run_id=run_id,
        details=generate_model_info()
    )
    val_records = ValidationRecords(db_path=TEST_PATH)
    val_records.create(
        participant_id=participant_id, 
        collab_id=collab_id, 
        project_id=project_id, 
        expt_id=expt_id, 
        run_id=run_id,
        details=generate_inference_info(action, 10, "evaluate")
    )
    pred_records = PredictionRecords(db_path=TEST_PATH)
    pred_records.create(
        participant_id=participant_id, 
        collab_id=collab_id, 
        project_id=project_id, 
        expt_id=expt_id, 
        run_id=run_id,
        details=generate_inference_info(action, 10, "predict")
    )

    return (
        project_records, 
        project_details,
        project_updates,
        (collab_id, project_id, expt_id, run_id, participant_id),
        (expt_records, run_records, model_records, val_records, pred_records)
    )


@pytest.fixture(scope='session')
def experiment_env():
    expt_records = ExperimentRecords(db_path=TEST_PATH)
    reset_database(expt_records)

    # Simulate data
    expt_details = generate_experiment_info() 
    expt_updates = generate_experiment_info()

    (collab_id, project_id, expt_id, run_id, participant_id
    ) = generate_federated_combination()

    action = random.choice(["classify", "regress"])

    # Generate downstream hierarchy
    run_records = RunRecords(db_path=TEST_PATH)
    run_records.create(
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id, 
        run_id=run_id,
        details=generate_run_info()
    )
    model_records = ModelRecords(db_path=TEST_PATH)
    model_records.create( 
        collab_id=collab_id, 
        project_id=project_id, 
        expt_id=expt_id, 
        run_id=run_id,
        details=generate_model_info()
    )
    val_records = ValidationRecords(db_path=TEST_PATH)
    val_records.create(
        participant_id=participant_id, 
        collab_id=collab_id, 
        project_id=project_id, 
        expt_id=expt_id, 
        run_id=run_id,
        details=generate_inference_info(action, 10, "evaluate")
    )
    pred_records = PredictionRecords(db_path=TEST_PATH)
    pred_records.create(
        participant_id=participant_id, 
        collab_id=collab_id, 
        project_id=project_id, 
        expt_id=expt_id, 
        run_id=run_id,
        details=generate_inference_info(action, 10, "predict")
    )

    return (
        expt_records, 
        expt_details,
        expt_updates,
        (collab_id, project_id, expt_id, run_id, participant_id),
        (run_records, model_records, val_records, pred_records)
    )


@pytest.fixture(scope='session')
def run_env():
    run_records = RunRecords(db_path=TEST_PATH)
    reset_database(run_records)

    # Simulate data
    run_details = generate_run_info() 
    run_updates = generate_run_info()

    (collab_id, project_id, expt_id, run_id, participant_id
    ) = generate_federated_combination()

    action = random.choice(["classify", "regress"])

    # Generate downstream hierarchy
    model_records = ModelRecords(db_path=TEST_PATH)
    model_records.create( 
        collab_id=collab_id, 
        project_id=project_id, 
        expt_id=expt_id, 
        run_id=run_id,
        details=generate_model_info()
    )
    val_records = ValidationRecords(db_path=TEST_PATH)
    val_records.create(
        participant_id=participant_id, 
        collab_id=collab_id, 
        project_id=project_id, 
        expt_id=expt_id, 
        run_id=run_id,
        details=generate_inference_info(action, 10, "evaluate")
    )
    pred_records = PredictionRecords(db_path=TEST_PATH)
    pred_records.create(
        participant_id=participant_id, 
        collab_id=collab_id, 
        project_id=project_id, 
        expt_id=expt_id, 
        run_id=run_id,
        details=generate_inference_info(action, 10, "predict")
    )

    return (
        run_records, 
        run_details,
        run_updates,
        (collab_id, project_id, expt_id, run_id, participant_id),
        (model_records, val_records, pred_records)
    )


@pytest.fixture(scope='session')
def model_env():
    model_records = ModelRecords(db_path=TEST_PATH)
    reset_database(model_records)

    # Simulate data
    model_details = generate_model_info() 
    model_updates = generate_model_info()

    (collab_id, project_id, expt_id, run_id, participant_id
    ) = generate_federated_combination()

    action = random.choice(["classify", "regress"])

    return (
        model_records, 
        model_details,
        model_updates,
        (collab_id, project_id, expt_id, run_id, participant_id)
    )


@pytest.fixture(scope='session')
def mlf_env():
    mlf_records = MLFRecords(db_path=TEST_PATH)
    reset_database(mlf_records)

    # Simulate data
    mlf_expt_details, mlf_run_details = generate_mlflow_info()
    mlf_expt_updates, mlf_run_updates = generate_mlflow_info() 

    (collab_id, project_id, expt_id, run_id, participant_id
    ) = generate_federated_combination()

    return (
        mlf_records, 
        (mlf_expt_details, mlf_run_details),
        (mlf_expt_updates, mlf_run_updates),
        (collab_id, project_id, expt_id, run_id, participant_id)
    )


@pytest.fixture(scope='session')
def validation_env():
    val_records = ValidationRecords(db_path=TEST_PATH)
    reset_database(val_records)

    # Simulate data
    action = random.choice(["classify", "regress"])
    validation_details = generate_inference_info(action, 10, "evaluate") 
    validation_updates = generate_inference_info(action, 10, "evaluate") 

    (collab_id, project_id, expt_id, run_id, participant_id
    ) = generate_federated_combination()

    return (
        val_records, 
        validation_details,
        validation_updates,
        (collab_id, project_id, expt_id, run_id, participant_id)
    )


@pytest.fixture(scope='session')
def prediction_env():
    pred_records = PredictionRecords(db_path=TEST_PATH)
    reset_database(pred_records)

    # Simulate data
    action = random.choice(["classify", "regress"])
    prediction_details = generate_inference_info(action, 10, "predict") 
    prediction_updates = generate_inference_info(action, 10, "predict") 

    (collab_id, project_id, expt_id, run_id, participant_id
    ) = generate_federated_combination()

    return (
        pred_records, 
        prediction_details,
        prediction_updates,
        (collab_id, project_id, expt_id, run_id, participant_id)
    )


@pytest.fixture(scope='session')
def participant_env():
    participant_records = ParticipantRecords(db_path=TEST_PATH)
    reset_database(participant_records)

    (collab_id, project_id, expt_id, run_id, participant_id
    ) = generate_federated_combination()

    # Simulate data
    participant_details = {'id': participant_id, **generate_participant_info()}
    participant_updates = generate_participant_info() 

    # Generate downstream hierarchy
    collaboration_records = CollaborationRecords(db_path=TEST_PATH)
    collaboration_records.create(
        collab_id=collab_id,
        details=generate_collaboration_info()
    )
    project_records = ProjectRecords(db_path=TEST_PATH)
    project_records.create(
        collab_id=collab_id,
        project_id=project_id,
        details=generate_project_info()
    )

    # Generate upstream hierarchy
    registration_records = RegistrationRecords(db_path=TEST_PATH)
    registration_records.create(
        collab_id=collab_id,
        project_id=project_id,
        participant_id=participant_id,
        details=generate_registration_info()
    )
    tag_records = TagRecords(db_path=TEST_PATH)
    tag_records.create( 
        collab_id=collab_id, 
        project_id=project_id,
        participant_id=participant_id, 
        details=generate_tag_info()
    )
    alignment_records = AlignmentRecords(db_path=TEST_PATH)
    alignment_records.create( 
        collab_id=collab_id, 
        project_id=project_id,
        participant_id=participant_id, 
        details=generate_alignment_info()
    )

    def reset_env():
        collaboration_records.delete(collab_id)

    return (
        participant_records, 
        participant_details,
        participant_updates,
        (collab_id, project_id, expt_id, run_id, participant_id),
        (registration_records, tag_records, alignment_records),
        reset_env
    )


@pytest.fixture(scope='session')
def registration_env():
    registration_records = RegistrationRecords(db_path=TEST_PATH)
    reset_database(registration_records)

    # Simulate data
    registration_details = generate_registration_info()
    registration_updates = generate_registration_info() 

    (collab_id, project_id, expt_id, run_id, participant_id
    ) = generate_federated_combination()

    # Generate downstream hierarchy
    collaboration_records = CollaborationRecords(db_path=TEST_PATH)
    collaboration_records.create(
        collab_id=collab_id,
        details=generate_collaboration_info()
    )
    project_records = ProjectRecords(db_path=TEST_PATH)
    project_records.create(
        collab_id=collab_id,
        project_id=project_id,
        details=generate_project_info()
    )
    participant_records = ParticipantRecords(db_path=TEST_PATH)
    participant_records.create(
        participant_id=participant_id,
        details={'id': participant_id, **generate_participant_info()}
    )

    # Generate upstream hierarchy
    tag_records = TagRecords(db_path=TEST_PATH)
    alignment_records = AlignmentRecords(db_path=TEST_PATH)
    
    def reset_env():
        collaboration_records.delete(collab_id)
        participant_records.delete(participant_id)

    return (
        registration_records, 
        registration_details,
        registration_updates,
        (collab_id, project_id, expt_id, run_id, participant_id),
        (tag_records, alignment_records),
        reset_env
    )


@pytest.fixture(scope='session')
def tag_env():
    tag_records = TagRecords(db_path=TEST_PATH)
    reset_database(tag_records)

    # Simulate data
    tag_details = generate_tag_info()
    tag_updates = generate_tag_info() 

    (collab_id, project_id, expt_id, run_id, participant_id
    ) = generate_federated_combination()

    # Generate downstream hierarchy
    collaboration_records = CollaborationRecords(db_path=TEST_PATH)
    collaboration_records.create(
        collab_id=collab_id,
        details=generate_collaboration_info()
    )
    project_records = ProjectRecords(db_path=TEST_PATH)
    project_records.create(
        collab_id=collab_id,
        project_id=project_id,
        details=generate_project_info()
    )
    participant_records = ParticipantRecords(db_path=TEST_PATH)
    participant_records.create(
        participant_id=participant_id,
        details={'id': participant_id, **generate_participant_info()}
    )

    # Generate upstream hierarchy
    registration_records = RegistrationRecords(db_path=TEST_PATH)
    registration_records.create( 
        collab_id=collab_id, 
        project_id=project_id,
        participant_id=participant_id, 
        details=generate_registration_info()
    )
    alignment_records = AlignmentRecords(db_path=TEST_PATH)

    def reset_env():
        collaboration_records.delete(collab_id)
        participant_records.delete(participant_id)

    return (
        tag_records, 
        tag_details,
        tag_updates,
        (collab_id, project_id, expt_id, run_id, participant_id),
        (registration_records, alignment_records),
        reset_env
    )


@pytest.fixture(scope='session')
def alignment_env():
    alignment_records = AlignmentRecords(db_path=TEST_PATH)
    reset_database(alignment_records)

    # Simulate data
    alignment_details = generate_alignment_info()
    alignment_updates = generate_alignment_info() 

    (collab_id, project_id, expt_id, run_id, participant_id
    ) = generate_federated_combination()

    # Generate upstream hierarchy
    registration_records = RegistrationRecords(db_path=TEST_PATH)
    registration_records.create( 
        collab_id=collab_id, 
        project_id=project_id,
        participant_id=participant_id, 
        details=generate_registration_info()
    )
    tag_records = TagRecords(db_path=TEST_PATH)
    tag_records.create(
        collab_id=collab_id, 
        project_id=project_id,
        participant_id=participant_id, 
        details=generate_tag_info()
    )

    def reset_env():
        registration_records.delete(collab_id, project_id, participant_id)

    return (
        alignment_records, 
        alignment_details,
        alignment_updates,
        (collab_id, project_id, expt_id, run_id, participant_id),
        reset_env
    )


if __name__ == "__main__":
    print(generate_registration_info())
    print(KEY_ID_MAPPINGS)
    print(LINK_ID_MAPPINGS)
    print(simulate_setup())
    print(generate_inference_info("classify", label_count=10, meta="evaluate"))
    print(generate_inference_info("regress", label_count=1, meta="predict"))