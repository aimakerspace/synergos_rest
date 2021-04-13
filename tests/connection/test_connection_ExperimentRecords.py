#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in


# Libs


# Custom
from conftest import (
    check_key_equivalence,
    check_relation_equivalence,
    check_detail_equivalence
)

##################
# Configurations #
##################


#################################
# ExperimentRecords Class Tests #
#################################

def test_ExperimentRecords_create(experiment_env):
    """ Tests if creation of experiment records is self-consistent and 
        hierarchy-enforcing.

    # C1: Check that specified record was dynamically created
    # C2: Check that specified record have a composite key
    # C3: Check that specified record was archived with correct substituent keys
    # C4: Check that specified record was archived with correct substituent IDs
    # C5: Check that specified record captured the correct specified details
    """
    (
        experiment_records, experiment_details, _,
        (collab_id, project_id, expt_id, _, _),
        _
    ) = experiment_env
    created_experiment = experiment_records.create(
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id, 
        details=experiment_details
    )
    # C1 - C4
    check_key_equivalence(
        record=created_experiment,
        ids=[collab_id, project_id, expt_id],
        r_type="experiment"
    )
    # C5
    check_detail_equivalence(
        record=created_experiment,
        details=experiment_details
    )


def test_ExperimentRecords_read_all(experiment_env):
    """ Tests if bulk reading of experiment records is self-consistent and 
        hierarchy-enforcing.

    # C1: Check that only 1 record exists (inherited from create())
    # C2: Check that specified record was dynamically created
    # C3: Check that specified record have a composite key
    # C4: Check that specified record was archived with correct substituent keys
    # C5: Check that specified record was archived with correct substituent IDs
    # C6: Check that specified record captured the correct specified details
    # C7: Check hierarchy-enforcing field "relations" exist
    # C8: Check that all downstream relations have been captured 
    """
    (
        experiment_records, experiment_details, _,
        (collab_id, project_id, expt_id, _, _),
        _
    ) = experiment_env
    all_experiments = experiment_records.read_all()
    # C1
    assert len(all_experiments) == 1
    for retrieved_record in all_experiments:
        # C2 - C5
        check_key_equivalence(
            record=retrieved_record,
            ids=[collab_id, project_id, expt_id],
            r_type="experiment"
        )
        # C6
        check_detail_equivalence(
            record=retrieved_record,
            details=experiment_details
        )
        # C7 - C8
        check_relation_equivalence(
            record=retrieved_record,
            r_type="experiment"
        )


def test_ExperimentRecords_read(experiment_env):
    """ Tests if single reading of experiment records is self-consistent and 
        hierarchy-enforcing.

    # C1: Check that specified record exists (inherited from create())
    # C2: Check that specified record was dynamically created
    # C3: Check that specified record have a composite key
    # C4: Check that specified record was archived with correct substituent keys
    # C5: Check that specified record was archived with correct substituent IDs
    # C6: Check that specified record captured the correct specified details
    # C7: Check hierarchy-enforcing field "relations" exist
    # C8: Check that all downstream relations have been captured 
    """
    (
        experiment_records, experiment_details, _,
        (collab_id, project_id, expt_id, _, _),
        _
    ) = experiment_env
    retrieved_experiment = experiment_records.read(
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id
    )
    # C1
    assert retrieved_experiment is not None
    # C2 - C5
    check_key_equivalence(
        record=retrieved_experiment,
        ids=[collab_id, project_id, expt_id],
        r_type="experiment"
    )
    # C6
    check_detail_equivalence(
        record=retrieved_experiment,
        details=experiment_details
    )
    # C7 - C8
    check_relation_equivalence(
        record=retrieved_experiment,
        r_type="experiment"
    )


def test_ExperimentRecords_update(experiment_env):
    """ Tests if a experiment record can be updated without breaking 
        hierarchial relations.

    # C1: Check that specified record was dynamically created
    # C2: Check that specified record has a composite key
    # C3: Check that specified record was archived with correct substituent keys
    # C4: Check that specified record was archived with correct substituent IDs
    # C5: Check that the original experiment record was updated (not a copy)
    # C6: Check that experiment record values have been updated
    # C7: Check hierarchy-enforcing field "relations" did not change
    """
    (
        experiment_records, _, experiment_updates,
        (collab_id, project_id, expt_id, _, _),
        _
    ) = experiment_env
    targeted_experiment = experiment_records.read(
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id
    )
    updated_experiment = experiment_records.update(
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id,
        updates=experiment_updates
    )
    retrieved_experiment = experiment_records.read(
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id
    )
    # C1 - C4
    check_key_equivalence(
        record=updated_experiment,
        ids=[collab_id, project_id, expt_id],
        r_type="experiment"
    )
    # C5
    assert targeted_experiment.doc_id == updated_experiment.doc_id
    # C6
    for k,v in experiment_updates.items():
        assert updated_experiment[k] == v  
    # C7
    assert targeted_experiment['relations'] == retrieved_experiment['relations']


def test_ExperimentRecords_delete(experiment_env):
    """ Tests if a experiment record can be deleted.

    # C1: Check that specified record was dynamically created
    # C2: Check that specified record has a composite key
    # C3: Check that specified record was archived with correct substituent keys
    # C4: Check that specified record was archived with correct substituent IDs
    # C5: Check that the original experiment record was deleted (not a copy)
    # C6: Check that specified experiment record no longer exists
    # C7: Check that all run records under current experiment no longer exists
    # C8: Check that all model records under current experiment no longer exists
    # C9: Check that all validation records under current experiment no longer exists
    # C10: Check that all prediction records under current experiment no longer exists
    """
    (
        experiment_records, _, _,
        (collab_id, project_id, expt_id, run_id, participant_id),
        (run_records, model_records, val_records, pred_records)
    ) = experiment_env
    targeted_experiment = experiment_records.read(
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id
    )
    deleted_experiment = experiment_records.delete(
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id
    )
    # C1 - C4
    check_key_equivalence(
        record=deleted_experiment,
        ids=[collab_id, project_id, expt_id],
        r_type="experiment"
    )
    # C5
    assert targeted_experiment.doc_id == deleted_experiment.doc_id
    # C6
    assert experiment_records.read(
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id
    ) is None
    # C7
    assert run_records.read(
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id
    ) is None
    # C8
    assert model_records.read(
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id
    ) is None
    # C9
    assert val_records.read(
        participant_id=participant_id,
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id
    ) is None
    # C10
    assert pred_records.read(
        participant_id=participant_id,
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id
    ) is None