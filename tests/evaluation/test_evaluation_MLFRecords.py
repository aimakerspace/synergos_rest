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
# MLFRecords Class Tests #
##########################

def test_MLFRecords_create(mlf_env):
    """ Tests if creation of mlf records is self-consistent and 
        hierarchy-enforcing.

    # C1: Check that specified record was dynamically created
    # C2: Check that specified record have a composite key
    # C3: Check that specified record was archived with correct substituent keys
    # C4: Check that specified record was archived with correct substituent IDs
    # C5: Check that specified record captured the correct specified details
    """
    (
        mlf_records, 
        all_mlf_details, 
        _,
        (collab_id, project_id, expt_id, run_id, _)
    ) = mlf_env

    relevant_ids = [expt_id, run_id]
    for mlf_details, record_id in zip(all_mlf_details, relevant_ids):
        created_mlf = mlf_records.create(
            collaboration=collab_id,
            project=project_id,
            name=record_id, 
            details=mlf_details
        )
        # C1 - C4
        check_key_equivalence(
            record=created_mlf,
            ids=[collab_id, project_id, record_id],
            r_type="mlflow"
        )
        # C5
        check_detail_equivalence(
            record=created_mlf,
            details=mlf_details
        )


def test_MLFRecords_read_all(mlf_env):
    """ Tests if bulk reading of mlf records is self-consistent and 
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
        mlf_records,
        (mlf_expt_details, mlf_run_details), 
        _,
        (collab_id, project_id, expt_id, run_id, _)
    ) = mlf_env
    all_mlfs = mlf_records.read_all()
    # C1
    assert len(all_mlfs) == 2
    for retrieved_record in all_mlfs:

        record_name = retrieved_record['key']['name']
        if record_name == expt_id:
            ids_test = [collab_id, project_id, expt_id]
            mlf_details = mlf_expt_details
        else:
            ids_test = [collab_id, project_id, run_id]
            mlf_details = mlf_run_details

        # C2 - C5
        check_key_equivalence(
            record=retrieved_record,
            ids=ids_test,
            r_type="mlflow"
        )
        # C6
        check_detail_equivalence(
            record=retrieved_record,
            details=mlf_details
        )
        # C7 - C8
        check_relation_equivalence(
            record=retrieved_record,
            r_type="mlflow"
        )


def test_MLFRecords_read(mlf_env):
    """ Tests if single reading of mlf records is self-consistent and 
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
        mlf_records, 
        all_mlf_details, 
        _,
        (collab_id, project_id, expt_id, run_id, _)
    ) = mlf_env

    relevant_ids = [expt_id, run_id]
    for mlf_details, record_id in zip(all_mlf_details, relevant_ids):
        retrieved_mlf = mlf_records.read(
            collaboration=collab_id,
            project=project_id,
            name=record_id
        )
        # C1
        assert retrieved_mlf is not None
        # C2 - C5
        check_key_equivalence(
            record=retrieved_mlf,
            ids=[collab_id, project_id, record_id],
            r_type="mlflow"
        )
        # C6
        check_detail_equivalence(
            record=retrieved_mlf,
            details=mlf_details
        )
        # C7 - C8
        check_relation_equivalence(
            record=retrieved_mlf,
            r_type="mlflow"
        )


def test_MLFRecords_update(mlf_env):
    """ Tests if a mlf record can be updated without breaking 
        hierarchial relations.

    # C1: Check that specified record was dynamically created
    # C2: Check that specified record has a composite key
    # C3: Check that specified record was archived with correct substituent keys
    # C4: Check that specified record was archived with correct substituent IDs
    # C5: Check that the original mlf record was updated (not a copy)
    # C6: Check that mlf record values have been updated
    # C7: Check hierarchy-enforcing field "relations" did not change
    """
    (
        mlf_records, 
        _, 
        all_mlf_updates,
        (collab_id, project_id, expt_id, run_id, _)
    ) = mlf_env

    relevant_ids = [expt_id, run_id]
    for mlf_updates, record_id in zip(all_mlf_updates, relevant_ids):
        targeted_mlf = mlf_records.read(
            collaboration=collab_id,
            project=project_id,
            name=record_id
        )
        updated_mlf = mlf_records.update(
            collaboration=collab_id,
            project=project_id,
            name=record_id,
            updates=mlf_updates
        )
        retrieved_mlf = mlf_records.read(
            collaboration=collab_id,
            project=project_id,
            name=record_id
        )
        # C1 - C4
        check_key_equivalence(
            record=updated_mlf,
            ids=[collab_id, project_id, record_id],
            r_type="mlflow"
        )
        # C5
        assert targeted_mlf.doc_id == updated_mlf.doc_id
        # C6
        for k,v in mlf_updates.items():
            assert updated_mlf[k] == v  
        # C7
        assert targeted_mlf['relations'] == retrieved_mlf['relations']


def test_MLFRecords_delete(mlf_env):
    """ Tests if a mlf record can be deleted.

    # C1: Check that specified record was dynamically created
    # C2: Check that specified record has a composite key
    # C3: Check that specified record was archived with correct substituent keys
    # C4: Check that specified record was archived with correct substituent IDs
    # C5: Check that the original mlf record was deleted (not a copy)
    # C6: Check that specified mlf record no longer exists
    """
    (
        mlf_records, _, _,
        (collab_id, project_id, expt_id, run_id, _)
    ) = mlf_env

    for record_id in (expt_id, run_id):
        targeted_mlf = mlf_records.read(
            collaboration=collab_id,
            project=project_id,
            name=record_id
        )
        deleted_mlf = mlf_records.delete(
            collaboration=collab_id,
            project=project_id,
            name=record_id
        )
        # C1 - C4
        check_key_equivalence(
            record=deleted_mlf,
            ids=[collab_id, project_id, record_id],
            r_type="mlflow"
        )
        # C5
        assert targeted_mlf.doc_id == deleted_mlf.doc_id
        # C6
        assert mlf_records.read(
            collaboration=collab_id,
            project=project_id,
            name=record_id
        ) is None