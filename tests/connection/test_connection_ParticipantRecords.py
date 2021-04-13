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


##################################
# ParticipantRecords Class Tests #
##################################

def test_ParticipantRecords_create(participant_env):
    """ Tests if creation of participant records is self-consistent and 
        hierarchy-enforcing.

    # C1: Check that specified record was dynamically created
    # C2: Check that specified record have a composite key
    # C3: Check that specified record was archived with correct substituent keys
    # C4: Check that specified record was archived with correct substituent IDs
    # C5: Check that specified record captured the correct specified details
    """
    (
        participant_records, participant_details, _,
        (_, _, _, _, participant_id),
        _, _
    ) = participant_env
    created_participant = participant_records.create(
        participant_id=participant_id,
        details=participant_details
    )
    # C1 - C4
    check_key_equivalence(
        record=created_participant,
        ids=[participant_id],
        r_type="participant"
    )
    # C5
    check_detail_equivalence(
        record=created_participant,
        details=participant_details
    )


def test_ParticipantRecords_read_all(participant_env):
    """ Tests if bulk reading of participant records is self-consistent and 
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
        participant_records, participant_details, _,
        (_, _, _, _, participant_id),
        _, _
    ) = participant_env
    all_participants = participant_records.read_all()
    # C1
    assert len(all_participants) == 1
    for retrieved_record in all_participants:
        # C2 - C5
        check_key_equivalence(
            record=retrieved_record,
            ids=[participant_id],
            r_type="participant"
        )
        # C6
        check_detail_equivalence(
            record=retrieved_record,
            details=participant_details
        )
        # C7 - C8
        check_relation_equivalence(
            record=retrieved_record,
            r_type="participant"
        )


def test_ParticipantRecords_read(participant_env):
    """ Tests if single reading of participant records is self-consistent and 
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
        participant_records, participant_details, _,
        (_, _, _, _, participant_id),
        _, _
    ) = participant_env
    retrieved_participant = participant_records.read(participant_id=participant_id)
    # C1
    assert retrieved_participant is not None
    # C2 - C5
    check_key_equivalence(
        record=retrieved_participant,
        ids=[participant_id],
        r_type="participant"
    )
    # C6
    check_detail_equivalence(
        record=retrieved_participant,
        details=participant_details
    )
    # C7 - C8
    check_relation_equivalence(
        record=retrieved_participant,
        r_type="participant"
    )


def test_ParticipantRecords_update(participant_env):
    """ Tests if a participant record can be updated without breaking 
        hierarchial relations.

    # C1: Check that specified record was dynamically created
    # C2: Check that specified record has a composite key
    # C3: Check that specified record was archived with correct substituent keys
    # C4: Check that specified record was archived with correct substituent IDs
    # C5: Check that the original participant record was updated (not a copy)
    # C6: Check that participant record values have been updated
    # C7: Check hierarchy-enforcing field "relations" did not change
    """
    (
        participant_records, _, participant_updates,
        (_, _, _, _, participant_id),
        _, _
    ) = participant_env
    targeted_participant = participant_records.read(participant_id=participant_id)
    updated_participant = participant_records.update(
        participant_id=participant_id,
        updates=participant_updates
    )
    retrieved_participant = participant_records.read(participant_id=participant_id)
    # C1 - C4
    check_key_equivalence(
        record=updated_participant,
        ids=[participant_id, participant_id],
        r_type="participant"
    )
    # C5
    assert targeted_participant.doc_id == updated_participant.doc_id
    # C6
    for k,v in participant_updates.items():
        assert updated_participant[k] == v  
    # C7
    assert targeted_participant['relations'] == retrieved_participant['relations']


def test_ParticipantRecords_delete(participant_env):
    """ Tests if a participant record can be deleted.

    # C1: Check that specified record was dynamically created
    # C2: Check that specified record has a composite key
    # C3: Check that specified record was archived with correct substituent keys
    # C4: Check that specified record was archived with correct substituent IDs
    # C5: Check that the original participant record was deleted (not a copy)
    # C6: Check that specified participant record no longer exists
    # C7: Check that all model records under current participant no longer exists
    # C8: Check that all validation records under current participant no longer exists
    # C9: Check that all prediction records under current participant no longer exists
    """
    (
        participant_records, _, _,
        (collab_id, project_id, _, _, participant_id),
        (registration_records, tag_records, alignment_records), reset_env
    ) = participant_env
    targeted_participant = participant_records.read(participant_id=participant_id)
    deleted_participant = participant_records.delete(participant_id=participant_id)
    # C1 - C4
    check_key_equivalence(
        record=deleted_participant,
        ids=[participant_id],
        r_type="participant"
    )
    # C5
    assert targeted_participant.doc_id == deleted_participant.doc_id
    # C6
    assert participant_records.read(participant_id=participant_id) is None
    # C7
    assert registration_records.read(
        collab_id=collab_id,
        project_id=project_id,
        participant_id=participant_id
    ) is None
    # C8
    assert tag_records.read(
        collab_id=collab_id,
        project_id=project_id,
        participant_id=participant_id
    ) is None    
    # C9
    assert alignment_records.read(
        collab_id=collab_id,
        project_id=project_id,
        participant_id=participant_id
    ) is None

    reset_env()