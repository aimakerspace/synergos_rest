#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in


# Libs


# Custom
from conftest import (
    generate_alignment_info,
    check_key_equivalence,
    check_relation_equivalence,
    check_link_equivalence,
    check_detail_equivalence
)


##################
# Configurations #
##################


##########################
# TagRecords Class Tests #
##########################

def test_TagRecords_create(tag_env):
    """ Tests if creation of tag records is self-consistent and 
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
        tag_records, tag_details, _,
        (collab_id, project_id, _, _, participant_id),
        _, _
    ) = tag_env
    created_tag = tag_records.create(
        collab_id=collab_id,
        project_id=project_id,
        participant_id=participant_id,
        details=tag_details
    )
    # C1 - C4
    check_key_equivalence(
        record=created_tag,
        ids=[participant_id, collab_id, project_id],
        r_type="tag"
    )
    # C5 - C6
    check_link_equivalence(
        record=created_tag,
        r_type="tag"
    )
    # C7
    check_detail_equivalence(
        record=created_tag,
        details=tag_details
    )


def test_TagRecords_read_all(tag_env):
    """ Tests if bulk reading of tag records is self-consistent and 
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
    # C11: Check that alignments captured have the correct details
    """
    (
        tag_records, tag_details, _,
        (collab_id, project_id, _, _, participant_id),
        (_, alignment_records), _
    ) = tag_env

    # Build downstream hierarchy 
    # (IMPT! Relations are only detected if records are created in sequence!)
    created_alignment = alignment_records.create( 
        collab_id=collab_id, 
        project_id=project_id,
        participant_id=participant_id, 
        details=generate_alignment_info()
    )

    all_tags = tag_records.read_all()
    # C1
    assert len(all_tags) == 1
    for retrieved_record in all_tags:
        # C2 - C5
        check_key_equivalence(
            record=retrieved_record,
            ids=[participant_id, collab_id, project_id],
            r_type="tag"
        )
        # C6 - C7
        check_link_equivalence(
            record=retrieved_record,
            r_type="tag"
        )
        # C8
        check_detail_equivalence(
            record=retrieved_record,
            details=tag_details
        )
        # C9 - C10
        check_relation_equivalence(
            record=retrieved_record,
            r_type="tag"
        )

        # C11
        related_alignment = retrieved_record['relations']['Alignment'][0]
        assert related_alignment == created_alignment

        # Clean up downstream hierarchy
        alignment_records.delete( 
            collab_id=collab_id, 
            project_id=project_id,
            participant_id=participant_id
        )


def test_TagRecords_read(tag_env):
    """ Tests if single reading of tag records is self-consistent and 
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
    # C11: Check that alignments captured have the correct details
    """
    (
        tag_records, tag_details, _,
        (collab_id, project_id, _, _, participant_id),
        (_, alignment_records), _
    ) = tag_env

    # Build downstream hierarchy 
    # (IMPT! Relations are only detected if records are created in sequence!)
    created_alignment = alignment_records.create( 
        collab_id=collab_id, 
        project_id=project_id,
        participant_id=participant_id, 
        details=generate_alignment_info()
    )

    retrieved_tag = tag_records.read(
        collab_id=collab_id,
        project_id=project_id,
        participant_id=participant_id
    )

    # C1
    assert retrieved_tag is not None
    # C2 - C5
    check_key_equivalence(
        record=retrieved_tag,
        ids=[participant_id, collab_id, project_id],
        r_type="tag"
    )
    # C6 - C7
    check_link_equivalence(
        record=retrieved_tag,
        r_type="tag"
    )
    # C8
    check_detail_equivalence(
        record=retrieved_tag,
        details=tag_details
    )
    # C9 - C10
    check_relation_equivalence(
        record=retrieved_tag,
        r_type="tag"
    )

    # C11
    related_alignment = retrieved_tag['relations']['Alignment'][0]
    assert related_alignment == created_alignment

    # Clean up downstream hierarchy
    alignment_records.delete( 
        collab_id=collab_id, 
        project_id=project_id,
        participant_id=participant_id
    )


def test_TagRecords_update(tag_env):
    """ Tests if a tag record can be updated without breaking 
        hierarchial relations.

    # C1: Check that specified record was dynamically created
    # C2: Check that specified record has a composite key
    # C3: Check that specified record was archived with correct substituent keys
    # C4: Check that specified record was archived with correct substituent IDs
    # C5: Check that composite key "link" exist for upstream transversal
    # C6: Check that keys in "link" are disjointed sets w.r.t "key"
    # C7: Check that the original tag record was updated (not a copy)
    # C8: Check that tag record values have been updated
    # C9: Check hierarchy-enforcing field "relations" did not change
    """
    (
        tag_records, _, tag_updates,
        (collab_id, project_id, _, _, participant_id),
        _, _
    ) = tag_env
    targeted_tag = tag_records.read(
        collab_id=collab_id,
        project_id=project_id,
        participant_id=participant_id
    )
    updated_tag = tag_records.update(
        collab_id=collab_id,
        project_id=project_id,
        participant_id=participant_id,
        updates=tag_updates
    )
    retrieved_tag = tag_records.read(
        collab_id=collab_id,
        project_id=project_id,
        participant_id=participant_id
    )
    # C1 - C4
    check_key_equivalence(
        record=updated_tag,
        ids=[participant_id, collab_id, project_id],
        r_type="tag"
    )
    # C5 - C6
    check_link_equivalence(
        record=updated_tag,
        r_type="tag"
    )
    # C7
    assert targeted_tag.doc_id == updated_tag.doc_id
    # C8
    for k,v in tag_updates.items():
        assert updated_tag[k] == v  
    # C9
    assert targeted_tag['relations'] == retrieved_tag['relations']


def test_TagRecords_delete(tag_env):
    """ Tests if a tag record can be deleted.

    # C1: Check that specified record was dynamically created
    # C2: Check that specified record has a composite key
    # C3: Check that specified record was archived with correct substituent keys
    # C4: Check that specified record was archived with correct substituent IDs
    # C5: Check that composite key "link" exist for upstream transversal
    # C6: Check that keys in "link" are disjointed sets w.r.t "key"
    # C7: Check that the original tag record was deleted (not a copy)
    # C8: Check that specified tag record no longer exists
    # C9: Check that all alignment records under current project no longer exists
    """
    (
        tag_records, _, _,
        (collab_id, project_id, _, _, participant_id),
        (_, alignment_records), reset_env
    ) = tag_env

    # Build remaining upstream hierarchy dynamically 
    # (IMPT! Relations are only detected if records are created in sequence!)
    alignment_records.create( 
        collab_id=collab_id, 
        project_id=project_id,
        participant_id=participant_id, 
        details=generate_alignment_info()
    )

    targeted_tag = tag_records.read(
        collab_id=collab_id,
        project_id=project_id,
        participant_id=participant_id
    )
    deleted_tag = tag_records.delete(
        collab_id=collab_id,
        project_id=project_id,
        participant_id=participant_id
    )
    # C1 - C4
    check_key_equivalence(
        record=deleted_tag,
        ids=[participant_id, collab_id, project_id],
        r_type="tag"
    )
    # C5 - C6
    check_link_equivalence(
        record=deleted_tag,
        r_type="tag"
    )
    # C7
    assert targeted_tag.doc_id == deleted_tag.doc_id
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
        participant_id=participant_id,
    ) is None

    reset_env()