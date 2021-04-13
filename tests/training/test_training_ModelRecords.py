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


############################
# ModelRecords Class Tests #
############################

def test_ModelRecords_create(model_env):
    """ Tests if creation of model records is self-consistent and 
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
        model_records, model_details, _,
        (collab_id, project_id, expt_id, run_id, _)
    ) = model_env
    created_model = model_records.create(
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id, 
        run_id=run_id,
        details=model_details
    )
    # C1 - C4
    check_key_equivalence(
        record=created_model,
        ids=[collab_id, project_id, expt_id, run_id],
        r_type="model"
    )
    # C5 - C6
    check_link_equivalence(
        record=created_model,
        r_type="model"
    )
    # C7
    check_detail_equivalence(
        record=created_model,
        details=model_details
    )


def test_ModelRecords_read_all(model_env):
    """ Tests if bulk reading of model records is self-consistent and 
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
        model_records, model_details, _,
        (collab_id, project_id, expt_id, run_id, _)
    ) = model_env
    all_models = model_records.read_all()
    # C1
    assert len(all_models) == 1
    for retrieved_record in all_models:
        # C2 - C5
        check_key_equivalence(
            record=retrieved_record,
            ids=[collab_id, project_id, expt_id, run_id],
            r_type="model"
        )
        # C6 - C7
        check_link_equivalence(
            record=retrieved_record,
            r_type="model"
        )
        # C8
        check_detail_equivalence(
            record=retrieved_record,
            details=model_details
        )
        # C9 - C10
        check_relation_equivalence(
            record=retrieved_record,
            r_type="model"
        )


def test_ModelRecords_read(model_env):
    """ Tests if single reading of model records is self-consistent and 
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
        model_records, model_details, _,
        (collab_id, project_id, expt_id, run_id, _)
    ) = model_env
    retrieved_model = model_records.read(
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id, 
        run_id=run_id,
    )
    # C1
    assert retrieved_model is not None
    # C2 - C5
    check_key_equivalence(
        record=retrieved_model,
        ids=[collab_id, project_id, expt_id, run_id],
        r_type="model"
    )
    # C6 - C7
    check_link_equivalence(
        record=retrieved_model,
        r_type="model"
    )
    # C8
    check_detail_equivalence(
        record=retrieved_model,
        details=model_details
    )
    # C9 - C10
    check_relation_equivalence(
        record=retrieved_model,
        r_type="model"
    )


def test_ModelRecords_update(model_env):
    """ Tests if a model record can be updated without breaking 
        hierarchial relations.

    # C1: Check that specified record was dynamically created
    # C2: Check that specified record has a composite key
    # C3: Check that specified record was archived with correct substituent keys
    # C4: Check that specified record was archived with correct substituent IDs
    # C5: Check that composite key "link" exist for upstream transversal
    # C6: Check that keys in "link" are disjointed sets w.r.t "key"
    # C7: Check that the original model record was updated (not a copy)
    # C8: Check that model record values have been updated
    # C9: Check hierarchy-enforcing field "relations" did not change
    """
    (
        model_records, _, model_updates,
        (collab_id, project_id, expt_id, run_id, _)
    ) = model_env
    targeted_model = model_records.read(
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id
    )
    updated_model = model_records.update(
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id,
        updates=model_updates
    )
    retrieved_model = model_records.read(
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id
    )
    # C1 - C4
    check_key_equivalence(
        record=updated_model,
        ids=[collab_id, project_id, expt_id, run_id],
        r_type="model"
    )
    # C5 - C6
    check_link_equivalence(
        record=updated_model,
        r_type="model"
    )
    # C7
    assert targeted_model.doc_id == updated_model.doc_id
    # C8
    for k,v in model_updates.items():
        assert updated_model[k] == v  
    # C9
    assert targeted_model['relations'] == retrieved_model['relations']


def test_ModelRecords_delete(model_env):
    """ Tests if a model record can be deleted.

    # C1: Check that specified record was dynamically created
    # C2: Check that specified record has a composite key
    # C3: Check that specified record was archived with correct substituent keys
    # C4: Check that specified record was archived with correct substituent IDs
    # C5: Check that composite key "link" exist for upstream transversal
    # C6: Check that keys in "link" are disjointed sets w.r.t "key"
    # C7: Check that the original model record was deleted (not a copy)
    # C8: Check that specified model record no longer exists
    """
    (
        model_records, _, _,
        (collab_id, project_id, expt_id, run_id, _)
    ) = model_env
    targeted_model = model_records.read(
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id
    )
    deleted_model = model_records.delete(
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id
    )
    # C1 - C4
    check_key_equivalence(
        record=deleted_model,
        ids=[collab_id, project_id, expt_id, run_id],
        r_type="model"
    )
    # C5 - C6
    check_link_equivalence(
        record=deleted_model,
        r_type="model"
    )
    # C7
    assert targeted_model.doc_id == deleted_model.doc_id
    # C8
    assert model_records.read(
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id
    ) is None