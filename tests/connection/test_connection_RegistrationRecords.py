#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import copy

# Libs
import tinydb

# Custom
from conftest import (
    generate_tag_info,
    generate_alignment_info,
    check_key_equivalence,
    check_relation_equivalence,
    check_link_equivalence,
    check_detail_equivalence
)

##################
# Configurations #
##################

def check_registration_detail_equivalence(
    record: tinydb.database.Document, 
    details: dict
) -> None:
    """ A registration entry is unconventional as compared to other records
        since it is an association record that pulls its downstream hierarchy
        into .read() and .read_all(). As such it has dynamically generated
        details. This function checks for this uniqueness.

    # C1: Check that collaboration details have been correctly imported
    # C2: Check that project details have been correctly imported
    # C3: Check that participant details have been correctly imported
    # C4: Check that specified record captured the correct specified details
    """
    # Ensure that a cloned record is no different from its original
    cloned_record = copy.deepcopy(record)
    assert cloned_record == record
    # C1
    involved_collaboration = cloned_record.pop('collaboration')
    assert involved_collaboration is not None
    # C2
    involved_project = cloned_record.pop('project')
    assert involved_project is not None
    # C3
    involved_participant = cloned_record.pop('participant')
    assert involved_participant is not None
    # C4
    check_detail_equivalence(record=cloned_record, details=details)

###################################
# RegistrationRecords Class Tests #
###################################

def test_RegistrationRecords_create(registration_env):
    """ Tests if creation of registration records is self-consistent and 
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
        registration_records, registration_details, _,
        (collab_id, project_id, _, _, participant_id),
        _, _
    ) = registration_env
    created_registration = registration_records.create(
        collab_id=collab_id,
        project_id=project_id,
        participant_id=participant_id,
        details=registration_details
    )
    # C1 - C4
    check_key_equivalence(
        record=created_registration,
        ids=[participant_id, collab_id, project_id],
        r_type="registration"
    )
    # C5 - C6
    check_link_equivalence(
        record=created_registration,
        r_type="registration"
    )
    # C7
    check_detail_equivalence(
        record=created_registration,
        details=registration_details
    )


def test_RegistrationRecords_read_all(registration_env):
    """ Tests if bulk reading of registration records is self-consistent and 
        hierarchy-enforcing.

    # C1: Check that only 1 record exists (inherited from create())
    # C2: Check that specified record was dynamically created
    # C3: Check that specified record have a composite key
    # C4: Check that specified record was archived with correct substituent keys
    # C5: Check that specified record was archived with correct substituent IDs
    # C6: Check that composite key "link" exist for upstream transversal
    # C7: Check that keys in "link" are disjointed sets w.r.t "key"
    # C8: Check that collaboration details have been correctly imported
    # C9: Check that project details have been correctly imported
    # C10: Check that participant details have been correctly imported
    # C11: Check that specified record captured the correct specified details
    # C12: Check hierarchy-enforcing field "relations" exist
    # C13: Check that all downstream relations have been captured 
    # C14: Check that tags captured have the correct details
    # C15: Check that alignments captured have the correct details
    """
    (
        registration_records, registration_details, _,
        (collab_id, project_id, _, _, participant_id),
        (tag_records, alignment_records), _
    ) = registration_env

    # Build remaining upstream hierarchy dynamically 
    # (IMPT! Relations are only detected if records are created in sequence!)
    created_tag = tag_records.create( 
        collab_id=collab_id, 
        project_id=project_id,
        participant_id=participant_id, 
        details=generate_tag_info()
    )
    created_alignment = alignment_records.create( 
        collab_id=collab_id, 
        project_id=project_id,
        participant_id=participant_id, 
        details=generate_alignment_info()
    )

    all_registrations = registration_records.read_all()
    # C1
    assert len(all_registrations) == 1
    for retrieved_record in all_registrations:
        # C2 - C5
        check_key_equivalence(
            record=retrieved_record,
            ids=[participant_id, collab_id, project_id],
            r_type="registration"
        )
        # C6 - C7
        check_link_equivalence(
            record=retrieved_record,
            r_type="registration"
        )
        # C8 -  C11
        check_registration_detail_equivalence(
            record=retrieved_record,
            details=registration_details
        )
        # C12 - C13
        check_relation_equivalence(
            record=retrieved_record,
            r_type="registration"
        )
        # C14
        related_tag = retrieved_record['relations']['Tag'][0]
        assert related_tag == created_tag
        # C15
        related_alignment = retrieved_record['relations']['Alignment'][0]
        assert related_alignment == created_alignment

        # Clean up downstream hierarchy
        alignment_records.delete( 
            collab_id=collab_id, 
            project_id=project_id,
            participant_id=participant_id
        )
        tag_records.delete( 
            collab_id=collab_id, 
            project_id=project_id,
            participant_id=participant_id
        )


def test_RegistrationRecords_read(registration_env):
    """ Tests if single reading of registration records is self-consistent and 
        hierarchy-enforcing.

    # C1: Check that specified record exists (inherited from create())
    # C2: Check that specified record was dynamically created
    # C3: Check that specified record have a composite key
    # C4: Check that specified record was archived with correct substituent keys
    # C5: Check that specified record was archived with correct substituent IDs
    # C6: Check that composite key "link" exist for upstream transversal
    # C7: Check that keys in "link" are disjointed sets w.r.t "key"
    # C8: Check that collaboration details have been correctly imported
    # C9: Check that project details have been correctly imported
    # C10: Check that participant details have been correctly imported
    # C11: Check that specified record captured the correct specified details
    # C12: Check hierarchy-enforcing field "relations" exist
    # C13: Check that all downstream relations have been captured 
    # C14: Check hierarchy-enforcing field "relations" exist
    # C15: Check that all downstream relations have been captured 
    """
    (
        registration_records, registration_details, _,
        (collab_id, project_id, _, _, participant_id),
        (tag_records, alignment_records), _
    ) = registration_env

    # Build remaining upstream hierarchy dynamically 
    # (IMPT! Relations are only detected if records are created in sequence!)
    created_tag = tag_records.create( 
        collab_id=collab_id, 
        project_id=project_id,
        participant_id=participant_id, 
        details=generate_tag_info()
    )
    created_alignment = alignment_records.create( 
        collab_id=collab_id, 
        project_id=project_id,
        participant_id=participant_id, 
        details=generate_alignment_info()
    )

    retrieved_registration = registration_records.read(
        collab_id=collab_id,
        project_id=project_id,
        participant_id=participant_id
    )

    # C1
    assert retrieved_registration is not None
    # C2 - C5
    check_key_equivalence(
        record=retrieved_registration,
        ids=[participant_id, collab_id, project_id],
        r_type="registration"
    )
    # C6 - C7
    check_link_equivalence(
        record=retrieved_registration,
        r_type="registration"
    )
    # C8 - C11
    check_registration_detail_equivalence(
        record=retrieved_registration,
        details=registration_details
    )
    # C12 - C13
    check_relation_equivalence(
        record=retrieved_registration,
        r_type="registration"
    )
    # C14
    related_tag = retrieved_registration['relations']['Tag'][0]
    assert related_tag == created_tag
    # C15
    related_alignment = retrieved_registration['relations']['Alignment'][0]
    assert related_alignment == created_alignment

    # Clean up downstream hierarchy
    alignment_records.delete( 
        collab_id=collab_id, 
        project_id=project_id,
        participant_id=participant_id
    )
    tag_records.delete( 
        collab_id=collab_id, 
        project_id=project_id,
        participant_id=participant_id
    )


def test_RegistrationRecords_update(registration_env):
    """ Tests if a registration record can be updated without breaking 
        hierarchial relations.

    # C1: Check that specified record was dynamically created
    # C2: Check that specified record has a composite key
    # C3: Check that specified record was archived with correct substituent keys
    # C4: Check that specified record was archived with correct substituent IDs
    # C5: Check that composite key "link" exist for upstream transversal
    # C6: Check that keys in "link" are disjointed sets w.r.t "key"
    # C7: Check that the original registration record was updated (not a copy)
    # C8: Check that registration record values have been updated
    # C9: Check hierarchy-enforcing field "relations" did not change
    """
    (
        registration_records, _, registration_updates,
        (collab_id, project_id, _, _, participant_id),
        _, _
    ) = registration_env
    targeted_registration = registration_records.read(
        collab_id=collab_id,
        project_id=project_id,
        participant_id=participant_id
    )
    updated_registration = registration_records.update(
        collab_id=collab_id,
        project_id=project_id,
        participant_id=participant_id,
        updates=registration_updates
    )
    retrieved_registration = registration_records.read(
        collab_id=collab_id,
        project_id=project_id,
        participant_id=participant_id
    )
    # C1 - C4
    check_key_equivalence(
        record=updated_registration,
        ids=[participant_id, collab_id, project_id],
        r_type="registration"
    )
    # C5 - C6
    check_link_equivalence(
        record=updated_registration,
        r_type="registration"
    )
    # C7
    assert targeted_registration.doc_id == updated_registration.doc_id
    # C8
    for k,v in registration_updates.items():
        assert updated_registration[k] == v  
    # C9
    assert targeted_registration['relations'] == retrieved_registration['relations']


def test_RegistrationRecords_delete(registration_env):
    """ Tests if a registration record can be deleted.

    # C1: Check that specified record was dynamically created
    # C2: Check that specified record has a composite key
    # C3: Check that specified record was archived with correct substituent keys
    # C4: Check that specified record was archived with correct substituent IDs
    # C5: Check that composite key "link" exist for upstream transversal
    # C6: Check that keys in "link" are disjointed sets w.r.t "key"
    # C7: Check that the original registration record was deleted (not a copy)
    # C8: Check that specified registration record no longer exists
    # C9: Check that all tag records under current project no longer exists
    # C10: Check that all alignment records under current project no longer exists
    """
    (
        registration_records, _, _,
        (collab_id, project_id, _, _, participant_id),
        (tag_records, alignment_records), reset_env
    ) = registration_env

    # Build remaining upstream hierarchy dynamically 
    # (IMPT! Relations are only detected if records are created in sequence!)
    tag_records.create( 
        collab_id=collab_id, 
        project_id=project_id,
        participant_id=participant_id, 
        details=generate_tag_info()
    )
    alignment_records.create( 
        collab_id=collab_id, 
        project_id=project_id,
        participant_id=participant_id, 
        details=generate_alignment_info()
    )

    targeted_registration = registration_records.read(
        collab_id=collab_id,
        project_id=project_id,
        participant_id=participant_id
    )
    deleted_registration = registration_records.delete(
        collab_id=collab_id,
        project_id=project_id,
        participant_id=participant_id
    )
    # C1 - C4
    check_key_equivalence(
        record=deleted_registration,
        ids=[participant_id, collab_id, project_id],
        r_type="registration"
    )
    # C5 - C6
    check_link_equivalence(
        record=deleted_registration,
        r_type="registration"
    )
    # C7
    assert targeted_registration.doc_id == deleted_registration.doc_id
    # C8
    assert registration_records.read(
        collab_id=collab_id,
        project_id=project_id,
        participant_id=participant_id
    ) is None
    # C9
    assert tag_records.read(
        collab_id=collab_id,
        project_id=project_id,
        participant_id=participant_id
    ) is None
    # C10
    assert alignment_records.read(
        participant_id=participant_id,
        collab_id=collab_id,
        project_id=project_id
    ) is None

    reset_env()