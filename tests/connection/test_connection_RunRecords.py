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


##########################
# RunRecords Class Tests #
##########################

def test_RunRecords_create(run_env):
    """ Tests if creation of run records is self-consistent and 
        hierarchy-enforcing.

    # C1: Check that specified record was dynamically created
    # C2: Check that specified record have a composite key
    # C3: Check that specified record was archived with correct substituent keys
    # C4: Check that specified record was archived with correct substituent IDsy"
    # C5: Check that specified record captured the correct specified details
    """
    (
        run_records, run_details, _,
        (collab_id, project_id, expt_id, run_id, _),
        _
    ) = run_env
    created_run = run_records.create(
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id, 
        run_id=run_id,
        details=run_details
    )
    # C1 - C4
    check_key_equivalence(
        record=created_run,
        ids=[collab_id, project_id, expt_id, run_id],
        r_type="run"
    )
    # C5
    check_detail_equivalence(
        record=created_run,
        details=run_details
    )


def test_RunRecords_read_all(run_env):
    """ Tests if bulk reading of run records is self-consistent and 
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
        run_records, run_details, _,
        (collab_id, project_id, expt_id, run_id, _),
        _
    ) = run_env
    all_runs = run_records.read_all()
    # C1
    assert len(all_runs) == 1
    for retrieved_record in all_runs:
        # C2 - C5
        check_key_equivalence(
            record=retrieved_record,
            ids=[collab_id, project_id, expt_id, run_id],
            r_type="run"
        )
        # C6
        check_detail_equivalence(
            record=retrieved_record,
            details=run_details
        )
        # C7 - C8
        check_relation_equivalence(
            record=retrieved_record,
            r_type="run"
        )


def test_RunRecords_read(run_env):
    """ Tests if single reading of run records is self-consistent and 
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
        run_records, run_details, _,
        (collab_id, project_id, expt_id, run_id, _),
        _
    ) = run_env
    retrieved_run = run_records.read(
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id, 
        run_id=run_id,
    )
    # C1
    assert retrieved_run is not None
    # C2 - C5
    check_key_equivalence(
        record=retrieved_run,
        ids=[collab_id, project_id, expt_id, run_id],
        r_type="run"
    )
    # C6
    check_detail_equivalence(
        record=retrieved_run,
        details=run_details
    )
    # C7 - C8
    check_relation_equivalence(
        record=retrieved_run,
        r_type="run"
    )


def test_RunRecords_update(run_env):
    """ Tests if a run record can be updated without breaking 
        hierarchial relations.

    # C1: Check that specified record was dynamically created
    # C2: Check that specified record has a composite key
    # C3: Check that specified record was archived with correct substituent keys
    # C4: Check that specified record was archived with correct substituent IDs
    # C5: Check that the original run record was updated (not a copy)
    # C6: Check that run record values have been updated
    # C7: Check hierarchy-enforcing field "relations" did not change
    """
    (
        run_records, _, run_updates,
        (collab_id, project_id, expt_id, run_id, _),
        _
    ) = run_env
    targeted_run = run_records.read(
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id
    )
    updated_run = run_records.update(
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id,
        updates=run_updates
    )
    retrieved_run = run_records.read(
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id
    )
    # C1 - C4
    check_key_equivalence(
        record=updated_run,
        ids=[collab_id, project_id, expt_id, run_id],
        r_type="run"
    )
    # C5
    assert targeted_run.doc_id == updated_run.doc_id
    # C6
    for k,v in run_updates.items():
        assert updated_run[k] == v  
    # C7
    assert targeted_run['relations'] == retrieved_run['relations']


def test_RunRecords_delete(run_env):
    """ Tests if a run record can be deleted.

    # C1: Check that specified record was dynamically created
    # C2: Check that specified record has a composite key
    # C3: Check that specified record was archived with correct substituent keys
    # C4: Check that specified record was archived with correct substituent IDs
    # C5: Check that the original run record was deleted (not a copy)
    # C6: Check that specified run record no longer exists
    # C7: Check that all model records under current run no longer exists
    # C8: Check that all validation records under current run no longer exists
    # C9: Check that all prediction records under current run no longer exists
    """
    (
        run_records, _, _,
        (collab_id, project_id, expt_id, run_id, participant_id),
        (model_records, val_records, pred_records)
    ) = run_env
    targeted_run = run_records.read(
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id
    )
    deleted_run = run_records.delete(
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id
    )
    # C1 - C4
    check_key_equivalence(
        record=deleted_run,
        ids=[collab_id, project_id, expt_id, run_id],
        r_type="run"
    )
    # C5
    assert targeted_run.doc_id == deleted_run.doc_id
    # C6
    assert run_records.read(
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id
    ) is None
    # C7
    assert model_records.read(
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id
    ) is None
    # C8
    assert val_records.read(
        participant_id=participant_id,
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id
    ) is None
    # C9
    assert pred_records.read(
        participant_id=participant_id,
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id
    ) is None