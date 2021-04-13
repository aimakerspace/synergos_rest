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
# ValidationRecords Class Tests #
#################################

def test_ValidationRecords_create(validation_env):
    """ Tests if creation of validation records is self-consistent and 
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
        validation_records, validation_details, _,
        (collab_id, project_id, expt_id, run_id, participant_id)
    ) = validation_env
    created_validation = validation_records.create(
        participant_id=participant_id,
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id, 
        run_id=run_id,
        details=validation_details
    )
    # C1 - C4
    check_key_equivalence(
        record=created_validation,
        ids=[participant_id, collab_id, project_id, expt_id, run_id],
        r_type="validation"
    )
    # C5 - C6
    check_link_equivalence(
        record=created_validation,
        r_type="validation"
    )
    # C7
    check_detail_equivalence(
        record=created_validation,
        details=validation_details
    )


def test_ValidationRecords_read_all(validation_env):
    """ Tests if bulk reading of validation records is self-consistent and 
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
        validation_records, validation_details, _,
        (collab_id, project_id, expt_id, run_id, participant_id)
    ) = validation_env
    all_validations = validation_records.read_all()
    # C1
    assert len(all_validations) == 1
    for retrieved_record in all_validations:
        # C2 - C5
        check_key_equivalence(
            record=retrieved_record,
            ids=[participant_id, collab_id, project_id, expt_id, run_id],
            r_type="validation"
        )
        # C6 - C7
        check_link_equivalence(
            record=retrieved_record,
            r_type="validation"
        )
        # C8
        check_detail_equivalence(
            record=retrieved_record,
            details=validation_details
        )
        # C9 - C10
        check_relation_equivalence(
            record=retrieved_record,
            r_type="validation"
        )


def test_ValidationRecords_read(validation_env):
    """ Tests if single reading of validation records is self-consistent and 
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
        validation_records, validation_details, _,
        (collab_id, project_id, expt_id, run_id, participant_id)
    ) = validation_env
    retrieved_validation = validation_records.read(
        participant_id=participant_id,
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id, 
        run_id=run_id,
    )
    # C1
    assert retrieved_validation is not None
    # C2 - C5
    check_key_equivalence(
        record=retrieved_validation,
        ids=[participant_id, collab_id, project_id, expt_id, run_id],
        r_type="validation"
    )
    # C6 - C7
    check_link_equivalence(
        record=retrieved_validation,
        r_type="validation"
    )
    # C8
    check_detail_equivalence(
        record=retrieved_validation,
        details=validation_details
    )
    # C9 - C10
    check_relation_equivalence(
        record=retrieved_validation,
        r_type="validation"
    )


def test_ValidationRecords_update(validation_env):
    """ Tests if a validation record can be updated without breaking 
        hierarchial relations.

    # C1: Check that specified record was dynamically created
    # C2: Check that specified record has a composite key
    # C3: Check that specified record was archived with correct substituent keys
    # C4: Check that specified record was archived with correct substituent IDs
    # C5: Check that composite key "link" exist for upstream transversal
    # C6: Check that keys in "link" are disjointed sets w.r.t "key"
    # C7: Check that the original validation record was updated (not a copy)
    # C8: Check that validation record values have been updated
    # C9: Check hierarchy-enforcing field "relations" did not change
    """
    (
        validation_records, _, validation_updates,
        (collab_id, project_id, expt_id, run_id, participant_id)
    ) = validation_env
    targeted_validation = validation_records.read(
        participant_id=participant_id,
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id
    )
    updated_validation = validation_records.update(
        participant_id=participant_id,
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id,
        updates=validation_updates
    )
    retrieved_validation = validation_records.read(
        participant_id=participant_id,
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id
    )
    # C1 - C4
    check_key_equivalence(
        record=updated_validation,
        ids=[participant_id, collab_id, project_id, expt_id, run_id],
        r_type="validation"
    )
    # C5 - C6
    check_link_equivalence(
        record=updated_validation,
        r_type="validation"
    )
    # C7
    assert targeted_validation.doc_id == updated_validation.doc_id
    # C8
    for k,v in validation_updates.items():
        assert updated_validation[k] == v  
    # C9
    assert targeted_validation['relations'] == retrieved_validation['relations']


def test_ValidationRecords_delete(validation_env):
    """ Tests if a validation record can be deleted.

    # C1: Check that specified record was dynamically created
    # C2: Check that specified record has a composite key
    # C3: Check that specified record was archived with correct substituent keys
    # C4: Check that specified record was archived with correct substituent IDs
    # C5: Check that composite key "link" exist for upstream transversal
    # C6: Check that keys in "link" are disjointed sets w.r.t "key"
    # C7: Check that the original validation record was deleted (not a copy)
    # C8: Check that specified validation record no longer exists
    """
    (
        validation_records, _, _,
        (collab_id, project_id, expt_id, run_id, participant_id)
    ) = validation_env
    targeted_validation = validation_records.read(
        participant_id=participant_id,
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id
    )
    deleted_validation = validation_records.delete(
        participant_id=participant_id,
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id
    )
    # C1 - C4
    check_key_equivalence(
        record=deleted_validation,
        ids=[participant_id, collab_id, project_id, expt_id, run_id],
        r_type="validation"
    )
    # C5 - C6
    check_link_equivalence(
        record=deleted_validation,
        r_type="validation"
    )
    # C7
    assert targeted_validation.doc_id == deleted_validation.doc_id
    # C8
    assert validation_records.read(
        participant_id=participant_id,
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id
    ) is None