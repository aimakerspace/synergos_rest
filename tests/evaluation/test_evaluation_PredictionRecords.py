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
    check_link_equivalence,
    check_detail_equivalence
)


##################
# Configurations #
##################


#################################
# PredictionRecords Class Tests #
#################################

def test_PredictionRecords_create(prediction_env):
    """ Tests if creation of prediction records is self-consistent and 
        hierarchy-enforcing.

    # C1: Check that specified record was dynamically created
    # C2: Check that specified record have a composite key
    # C3: Check that specified record was archived with correct substituent keys
    # C4: Check that specified record was archived with correct substituent IDs
    # C5: Check that composite key "link" exist for upstream transversal
    # C6: Check that keys in "link" are disjointed sets w.r.t "key"
    # C7: Check that specified record captured the correct specified details
    """
    (
        prediction_records, prediction_details, _,
        (collab_id, project_id, expt_id, run_id, participant_id)
    ) = prediction_env
    created_prediction = prediction_records.create(
        participant_id=participant_id,
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id, 
        run_id=run_id,
        details=prediction_details
    )
    # C1 - C4
    check_key_equivalence(
        record=created_prediction,
        ids=[participant_id, collab_id, project_id, expt_id, run_id],
        r_type="prediction"
    )
    # C5 - C6
    check_link_equivalence(
        record=created_prediction,
        r_type="prediction"
    )
    # C7
    check_detail_equivalence(
        record=created_prediction,
        details=prediction_details
    )


def test_PredictionRecords_read_all(prediction_env):
    """ Tests if bulk reading of prediction records is self-consistent and 
        hierarchy-enforcing.

    # C1: Check that only 1 record exists (inherited from create())
    # C2: Check that specified record was dynamically created
    # C3: Check that specified record have a composite key
    # C4: Check that specified record was archived with correct substituent keys
    # C5: Check that specified record was archived with correct substituent IDs
    # C6: Check that composite key "link" exist for upstream transversal
    # C7: Check that keys in "link" are disjointed sets w.r.t "key"
    # C8: Check that specified record captured the correct specified details
    # C9: Check hierarchy-enforcing field "relations" exist
    # C10: Check that all downstream relations have been captured 
    """
    (
        prediction_records, prediction_details, _,
        (collab_id, project_id, expt_id, run_id, participant_id)
    ) = prediction_env
    all_predictions = prediction_records.read_all()
    # C1
    assert len(all_predictions) == 1
    for retrieved_record in all_predictions:
        # C2 - C5
        check_key_equivalence(
            record=retrieved_record,
            ids=[participant_id, collab_id, project_id, expt_id, run_id],
            r_type="prediction"
        )
        # C6 - C7
        check_link_equivalence(
            record=retrieved_record,
            r_type="prediction"
        )
        # C8
        check_detail_equivalence(
            record=retrieved_record,
            details=prediction_details
        )
        # C9 - C10
        check_relation_equivalence(
            record=retrieved_record,
            r_type="prediction"
        )


def test_PredictionRecords_read(prediction_env):
    """ Tests if single reading of prediction records is self-consistent and 
        hierarchy-enforcing.

    # C1: Check that specified record exists (inherited from create())
    # C2: Check that specified record was dynamically created
    # C3: Check that specified record have a composite key
    # C4: Check that specified record was archived with correct substituent keys
    # C5: Check that specified record was archived with correct substituent IDs
    # C6: Check that composite key "link" exist for upstream transversal
    # C7: Check that keys in "link" are disjointed sets w.r.t "key"
    # C8: Check that specified record captured the correct specified details
    # C9: Check hierarchy-enforcing field "relations" exist
    # C10: Check that all downstream relations have been captured 
    """
    (
        prediction_records, prediction_details, _,
        (collab_id, project_id, expt_id, run_id, participant_id)
    ) = prediction_env
    retrieved_prediction = prediction_records.read(
        participant_id=participant_id,
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id, 
        run_id=run_id,
    )
    # C1
    assert retrieved_prediction is not None
    # C2 - C5
    check_key_equivalence(
        record=retrieved_prediction,
        ids=[participant_id, collab_id, project_id, expt_id, run_id],
        r_type="prediction"
    )
    # C6 - C7
    check_link_equivalence(
        record=retrieved_prediction,
        r_type="prediction"
    )
    # C8
    check_detail_equivalence(
        record=retrieved_prediction,
        details=prediction_details
    )
    # C9 - C10
    check_relation_equivalence(
        record=retrieved_prediction,
        r_type="prediction"
    )


def test_PredictionRecords_update(prediction_env):
    """ Tests if a prediction record can be updated without breaking 
        hierarchial relations.

    # C1: Check that specified record was dynamically created
    # C2: Check that specified record has a composite key
    # C3: Check that specified record was archived with correct substituent keys
    # C4: Check that specified record was archived with correct substituent IDs
    # C5: Check that composite key "link" exist for upstream transversal
    # C6: Check that keys in "link" are disjointed sets w.r.t "key"
    # C7: Check that the original prediction record was updated (not a copy)
    # C8: Check that prediction record values have been updated
    # C9: Check hierarchy-enforcing field "relations" did not change
    """
    (
        prediction_records, _, prediction_updates,
        (collab_id, project_id, expt_id, run_id, participant_id)
    ) = prediction_env
    targeted_prediction = prediction_records.read(
        participant_id=participant_id,
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id
    )
    updated_prediction = prediction_records.update(
        participant_id=participant_id,
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id,
        updates=prediction_updates
    )
    retrieved_prediction = prediction_records.read(
        participant_id=participant_id,
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id
    )
    # C1 - C4
    check_key_equivalence(
        record=updated_prediction,
        ids=[participant_id, collab_id, project_id, expt_id, run_id],
        r_type="prediction"
    )
    # C5 - C6
    check_link_equivalence(
        record=updated_prediction,
        r_type="prediction"
    )
    # C7
    assert targeted_prediction.doc_id == updated_prediction.doc_id
    # C8
    for k,v in prediction_updates.items():
        assert updated_prediction[k] == v  
    # C9
    assert targeted_prediction['relations'] == retrieved_prediction['relations']


def test_PredictionRecords_delete(prediction_env):
    """ Tests if a prediction record can be deleted.

    # C1: Check that specified record was dynamically created
    # C2: Check that specified record has a composite key
    # C3: Check that specified record was archived with correct substituent keys
    # C4: Check that specified record was archived with correct substituent IDs
    # C5: Check that composite key "link" exist for upstream transversal
    # C6: Check that keys in "link" are disjointed sets w.r.t "key"
    # C7: Check that the original prediction record was deleted (not a copy)
    # C8: Check that specified prediction record no longer exists
    """
    (
        prediction_records, _, _,
        (collab_id, project_id, expt_id, run_id, participant_id)
    ) = prediction_env
    targeted_prediction = prediction_records.read(
        participant_id=participant_id,
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id
    )
    deleted_prediction = prediction_records.delete(
        participant_id=participant_id,
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id
    )
    # C1 - C4
    check_key_equivalence(
        record=deleted_prediction,
        ids=[participant_id, collab_id, project_id, expt_id, run_id],
        r_type="prediction"
    )
    # C5 - C6
    check_link_equivalence(
        record=deleted_prediction,
        r_type="prediction"
    )
    # C7
    assert targeted_prediction.doc_id == deleted_prediction.doc_id
    # C8
    assert prediction_records.read(
        participant_id=participant_id,
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id
    ) is None