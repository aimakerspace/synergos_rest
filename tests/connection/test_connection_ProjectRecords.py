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


##############################
# ProjectRecords Class Tests #
##############################

def test_ProjectRecords_create(project_env):
    """ Tests if creation of project records is self-consistent and 
        hierarchy-enforcing.

    # C1: Check that specified record was dynamically created
    # C2: Check that specified record have a composite key
    # C3: Check that specified record was archived with correct substituent keys
    # C4: Check that specified record was archived with correct substituent IDs
    # C5: Check that specified record captured the correct specified details
    """
    (
        project_records, project_details, _,
        (collab_id, project_id, _, _, _),
        _
    ) = project_env
    created_project = project_records.create(
        collab_id=collab_id,
        project_id=project_id,
        details=project_details
    )
    # C1 - C4
    check_key_equivalence(
        record=created_project,
        ids=[collab_id, project_id],
        r_type="project"
    )
    # C5
    check_detail_equivalence(
        record=created_project,
        details=project_details
    )


def test_ProjectRecords_read_all(project_env):
    """ Tests if bulk reading of project records is self-consistent and 
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
        project_records, project_details, _,
        (collab_id, project_id, _, _, _),
        _
    ) = project_env
    all_projects = project_records.read_all()
    # C1
    assert len(all_projects) == 1
    for retrieved_record in all_projects:
        # C2 - C5
        check_key_equivalence(
            record=retrieved_record,
            ids=[collab_id, project_id],
            r_type="project"
        )
        # C6
        check_detail_equivalence(
            record=retrieved_record,
            details=project_details
        )
        # C7 - C8
        check_relation_equivalence(
            record=retrieved_record,
            r_type="project"
        )


def test_ProjectRecords_read(project_env):
    """ Tests if single reading of project records is self-consistent and 
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
        project_records, project_details, _,
        (collab_id, project_id, _, _, _),
        _
    ) = project_env
    retrieved_project = project_records.read(
        collab_id=collab_id,
        project_id=project_id
    )
    # C1
    assert retrieved_project is not None
    # C2 - C5
    check_key_equivalence(
        record=retrieved_project,
        ids=[collab_id, project_id],
        r_type="project"
    )
    # C6
    check_detail_equivalence(
        record=retrieved_project,
        details=project_details
    )
    # C7 - C8
    check_relation_equivalence(
        record=retrieved_project,
        r_type="project"
    )


def test_ProjectRecords_update(project_env):
    """ Tests if a project record can be updated without breaking 
        hierarchial relations.

    # C1: Check that specified record was dynamically created
    # C2: Check that specified record has a composite key
    # C3: Check that specified record was archived with correct substituent keys
    # C4: Check that specified record was archived with correct substituent IDs
    # C5: Check that the original project record was updated (not a copy)
    # C6: Check that project record values have been updated
    # C7: Check hierarchy-enforcing field "relations" did not change
    """
    (
        project_records, _, project_updates,
        (collab_id, project_id, _, _, _),
        _
    ) = project_env
    targeted_project = project_records.read(
        collab_id=collab_id,
        project_id=project_id
    )
    updated_project = project_records.update(
        collab_id=collab_id,
        project_id=project_id,
        updates=project_updates
    )
    retrieved_project = project_records.read(
        collab_id=collab_id,
        project_id=project_id
    )
    # C1 - C4
    check_key_equivalence(
        record=updated_project,
        ids=[collab_id, project_id],
        r_type="project"
    )
    # C5
    assert targeted_project.doc_id == updated_project.doc_id
    # C6
    for k,v in project_updates.items():
        assert updated_project[k] == v  
    # C7
    assert targeted_project['relations'] == retrieved_project['relations']


def test_ProjectRecords_delete(project_env):
    """ Tests if a project record can be deleted.

    # C1: Check that specified record was dynamically created
    # C2: Check that specified record has a composite key
    # C3: Check that specified record was archived with correct substituent keys
    # C4: Check that specified record was archived with correct substituent IDs
    # C5: Check that the original project record was deleted (not a copy)
    # C6: Check that specified project record no longer exists
    # C7: Check that all experiment records under current project no longer exists
    # C8: Check that all run records under current project no longer exists
    # C9: Check that all model records under current project no longer exists
    # C10: Check that all validation records under current project no longer exists
    # C11: Check that all prediction records under current project no longer exists
    """
    (
        project_records, _, _,
        (collab_id, project_id, expt_id, run_id, participant_id),
        (expt_records, run_records, model_records, val_records, pred_records)
    ) = project_env
    targeted_project = project_records.read(
        collab_id=collab_id,
        project_id=project_id
    )
    deleted_project = project_records.delete(
        collab_id=collab_id,
        project_id=project_id
    )
    # C1 - C4
    check_key_equivalence(
        record=deleted_project,
        ids=[collab_id, project_id],
        r_type="project"
    )
    # C5
    assert targeted_project.doc_id == deleted_project.doc_id
    # C6
    assert project_records.read(
        collab_id=collab_id,
        project_id=project_id
    ) is None
    # C7
    assert expt_records.read(
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id
    ) is None    
    # C8
    assert run_records.read(
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id
    ) is None
    # C9
    assert model_records.read(
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id
    ) is None
    # C10
    assert val_records.read(
        participant_id=participant_id,
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id
    ) is None
    # C11
    assert pred_records.read(
        participant_id=participant_id,
        collab_id=collab_id,
        project_id=project_id,
        expt_id=expt_id,
        run_id=run_id
    ) is None