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


####################################
# CollaborationRecords Class Tests #
####################################

def test_CollaborationRecords_create(collab_env):
    """ Tests if creation of collab records is self-consistent and 
        hierarchy-enforcing.

    # C1: Check that specified record was dynamically created
    # C2: Check that specified record have a composite key
    # C3: Check that specified record was archived with correct substituent keys
    # C4: Check that specified record was archived with correct substituent IDs
    # C5: Check that specified record captured the correct specified details
    """
    (
        collab_records, collab_details, _,
        (collab_id, _, _, _, _),
        _, 
    ) = collab_env
    created_collab = collab_records.create(
        collab_id=collab_id,
        details=collab_details
    )
    # C1 - C4
    check_key_equivalence(
        record=created_collab,
        ids=[collab_id],
        r_type="collaboration"
    )
    # C5
    check_detail_equivalence(
        record=created_collab,
        details=collab_details
    )


def test_CollaborationRecords_read_all(collab_env):
    """ Tests if bulk reading of collab records is self-consistent and 
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
        collab_records, collab_details, _,
        (collab_id, _, _, _, _),
        _, 
    ) = collab_env
    all_collabs = collab_records.read_all()
    # C1
    assert len(all_collabs) == 1
    for retrieved_record in all_collabs:
        # C2 - C5
        check_key_equivalence(
            record=retrieved_record,
            ids=[collab_id],
            r_type="collaboration"
        )
        # C6
        check_detail_equivalence(
            record=retrieved_record,
            details=collab_details
        )
        # C7 - C8
        check_relation_equivalence(
            record=retrieved_record,
            r_type="collaboration"
        )


def test_CollaborationRecords_read(collab_env):
    """ Tests if single reading of collab records is self-consistent and 
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
        collab_records, collab_details, _,
        (collab_id, _, _, _, _),
        _, 
    ) = collab_env
    retrieved_collab = collab_records.read(collab_id=collab_id)
    # C1
    assert retrieved_collab is not None
    # C2 - C5
    check_key_equivalence(
        record=retrieved_collab,
        ids=[collab_id],
        r_type="collaboration"
    )
    # C6
    check_detail_equivalence(
        record=retrieved_collab,
        details=collab_details
    )
    # C7 - C8
    check_relation_equivalence(
        record=retrieved_collab,
        r_type="collaboration"
    )


def test_CollaborationRecords_update(collab_env):
    """ Tests if a collab record can be updated without breaking 
        hierarchial relations.

    # C1: Check that specified record was dynamically created
    # C2: Check that specified record has a composite key
    # C3: Check that specified record was archived with correct substituent keys
    # C4: Check that specified record was archived with correct substituent IDs
    # C5: Check that the original collab record was updated (not a copy)
    # C6: Check that collab record values have been updated
    # C7: Check hierarchy-enforcing field "relations" did not change
    """
    (
        collab_records, _, collab_updates,
        (collab_id, _, _, _, _),
        _, 
    ) = collab_env
    targeted_collab = collab_records.read(collab_id=collab_id)
    updated_collab = collab_records.update(
        collab_id=collab_id,
        updates=collab_updates
    )
    retrieved_collab = collab_records.read(collab_id=collab_id)
    # C1 - C4
    check_key_equivalence(
        record=updated_collab,
        ids=[collab_id, collab_id],
        r_type="collaboration"
    )
    # C5
    assert targeted_collab.doc_id == updated_collab.doc_id
    # C6
    for k,v in collab_updates.items():
        assert updated_collab[k] == v  
    # C7
    assert targeted_collab['relations'] == retrieved_collab['relations']


def test_CollaborationRecords_delete(collab_env):
    """ Tests if a collab record can be deleted.

    # C1: Check that specified record was dynamically created
    # C2: Check that specified record has a composite key
    # C3: Check that specified record was archived with correct substituent keys
    # C4: Check that specified record was archived with correct substituent IDs
    # C5: Check that the original collab record was deleted (not a copy)
    # C6: Check that specified collab record no longer exists
    # C7: Check that all project records under current collab no longer exists
    # C8: Check that all experiment records under current collab no longer exists
    # C9: Check that all run records under current collab no longer exists
    # C10: Check that all model records under current collab no longer exists
    # C11: Check that all validation records under current collab no longer exists
    # C12: Check that all prediction records under current collab no longer exists
    # C13: Check that all registration records under current collab no longer exists
    # C14: Check that all tag records under current collab no longer exists
    # C15: Check that all alignment records under current collab no longer exists
    """
    (
        collab_records, _, _,
        (collab_id, project_id, expt_id, run_id, participant_id),
        (project_records, expt_records, run_records, model_records, 
         val_records, pred_records,
         registration_records, tag_records, alignment_records)
    ) = collab_env
    targeted_collab = collab_records.read(collab_id=collab_id)
    deleted_collab = collab_records.delete(collab_id=collab_id)
    # C1 - C4
    check_key_equivalence(
        record=deleted_collab,
        ids=[collab_id],
        r_type="collaboration"
    )
    # C5
    assert targeted_collab.doc_id == deleted_collab.doc_id
    # C6
    assert collab_records.read(collab_id=collab_id) is None
    # C7
    assert project_records.read(
        collab_id=collab_id,
        project_id=project_id
    ) is None
    # C8
    assert expt_records.read(
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id
    ) is None    
    # C9
    assert run_records.read(
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id
    ) is None
    # C10
    assert model_records.read(
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id
    ) is None
    # C11
    assert val_records.read(
        participant_id=participant_id,
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id
    ) is None
    # C12
    assert pred_records.read(
        participant_id=participant_id,
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id
    ) is None
    # C13
    assert registration_records.read(
        collab_id=collab_id,
        project_id=project_id,
        participant_id=participant_id
    ) is None
    # C14
    assert tag_records.read(
        collab_id=collab_id,
        project_id=project_id,
        participant_id=participant_id
    ) is None
    # C15
    assert alignment_records.read(
        collab_id=collab_id,
        project_id=project_id,
        participant_id=participant_id
    ) is None