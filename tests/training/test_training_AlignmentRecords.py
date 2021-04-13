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


################################
# AlignmentRecords Class Tests #
################################

def test_AlignmentRecords_create(alignment_env):
    """ Tests if creation of alignment records is self-consistent and 
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
        alignment_records, alignment_details, _,
        (collab_id, project_id, _, _, participant_id), _
    ) = alignment_env
    created_alignment = alignment_records.create(
        collab_id=collab_id,
        project_id=project_id,
        participant_id=participant_id,
        details=alignment_details
    )
    # C1 - C4
    check_key_equivalence(
        record=created_alignment,
        ids=[participant_id, collab_id, project_id],
        r_type="alignment"
    )
    # C5 - C6
    check_link_equivalence(
        record=created_alignment,
        r_type="alignment"
    )
    # C7
    check_detail_equivalence(
        record=created_alignment,
        details=alignment_details
    )


def test_AlignmentRecords_read_all(alignment_env):
    """ Tests if bulk reading of alignment records is self-consistent and 
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
        alignment_records, alignment_details, _,
        (collab_id, project_id, _, _, participant_id), _
    ) = alignment_env
    all_alignments = alignment_records.read_all()
    # C1
    assert len(all_alignments) == 1
    for retrieved_record in all_alignments:
        # C2 - C5
        check_key_equivalence(
            record=retrieved_record,
            ids=[participant_id, collab_id, project_id],
            r_type="alignment"
        )
        # C6 - C7
        check_link_equivalence(
            record=retrieved_record,
            r_type="alignment"
        )
        # C8
        check_detail_equivalence(
            record=retrieved_record,
            details=alignment_details
        )
        # C9 - C10
        check_relation_equivalence(
            record=retrieved_record,
            r_type="alignment"
        )


def test_AlignmentRecords_read(alignment_env):
    """ Tests if single reading of alignment records is self-consistent and 
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
        alignment_records, alignment_details, _,
        (collab_id, project_id, _, _, participant_id), _
    ) = alignment_env
    retrieved_alignment = alignment_records.read(
        collab_id=collab_id,
        project_id=project_id,
        participant_id=participant_id
    )
    # C1
    assert retrieved_alignment is not None
    # C2 - C5
    check_key_equivalence(
        record=retrieved_alignment,
        ids=[participant_id, collab_id, project_id],
        r_type="alignment"
    )
    # C6 - C7
    check_link_equivalence(
        record=retrieved_alignment,
        r_type="alignment"
    )
    # C8
    check_detail_equivalence(
        record=retrieved_alignment,
        details=alignment_details
    )
    # C9 - C10
    check_relation_equivalence(
        record=retrieved_alignment,
        r_type="alignment"
    )


def test_AlignmentRecords_update(alignment_env):
    """ Tests if a alignment record can be updated without breaking 
        hierarchial relations.

    # C1: Check that specified record was dynamically created
    # C2: Check that specified record has a composite key
    # C3: Check that specified record was archived with correct substituent keys
    # C4: Check that specified record was archived with correct substituent IDs
    # C5: Check that composite key "link" exist for upstream transversal
    # C6: Check that keys in "link" are disjointed sets w.r.t "key"
    # C7: Check that the original alignment record was updated (not a copy)
    # C8: Check that alignment record values have been updated
    # C9: Check hierarchy-enforcing field "relations" did not change
    """
    (
        alignment_records, _, alignment_updates,
        (collab_id, project_id, _, _, participant_id), _
    ) = alignment_env
    targeted_alignment = alignment_records.read(
        collab_id=collab_id,
        project_id=project_id,
        participant_id=participant_id
    )
    updated_alignment = alignment_records.update(
        collab_id=collab_id,
        project_id=project_id,
        participant_id=participant_id,
        updates=alignment_updates
    )
    retrieved_alignment = alignment_records.read(
        collab_id=collab_id,
        project_id=project_id,
        participant_id=participant_id
    )
    # C1 - C4
    check_key_equivalence(
        record=updated_alignment,
        ids=[participant_id, collab_id, project_id],
        r_type="alignment"
    )
    # C5 - C6
    check_link_equivalence(
        record=updated_alignment,
        r_type="alignment"
    )
    # C7
    assert targeted_alignment.doc_id == updated_alignment.doc_id
    # C8
    for k,v in alignment_updates.items():
        assert updated_alignment[k] == v  
    # C9
    assert targeted_alignment['relations'] == retrieved_alignment['relations']


def test_AlignmentRecords_delete(alignment_env):
    """ Tests if a alignment record can be deleted.

    # C1: Check that specified record was dynamically created
    # C2: Check that specified record has a composite key
    # C3: Check that specified record was archived with correct substituent keys
    # C4: Check that specified record was archived with correct substituent IDs
    # C5: Check that composite key "link" exist for upstream transversal
    # C6: Check that keys in "link" are disjointed sets w.r.t "key"
    # C7: Check that the original alignment record was deleted (not a copy)
    # C8: Check that specified alignment record no longer exists
    """
    (
        alignment_records, _, _,
        (collab_id, project_id, _, _, participant_id), reset_env
    ) = alignment_env
    targeted_alignment = alignment_records.read(
        collab_id=collab_id,
        project_id=project_id,
        participant_id=participant_id
    )
    deleted_alignment = alignment_records.delete(
        collab_id=collab_id,
        project_id=project_id,
        participant_id=participant_id
    )
    # C1 - C4
    check_key_equivalence(
        record=deleted_alignment,
        ids=[participant_id, collab_id, project_id],
        r_type="alignment"
    )
    # C5 - C6
    check_link_equivalence(
        record=deleted_alignment,
        r_type="alignment"
    )
    # C7
    assert targeted_alignment.doc_id == deleted_alignment.doc_id
    # C8
    assert alignment_records.read(
        collab_id=collab_id,
        project_id=project_id,
        participant_id=participant_id
    ) is None

    reset_env()