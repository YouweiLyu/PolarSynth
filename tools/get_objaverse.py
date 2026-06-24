import os
import objaverse.xl as oxl
from typing import Dict, Hashable, Any

def handle_found_object(
    local_path: str,
    file_identifier: str,
    sha256: str,
    metadata: Dict[Hashable, Any]
) -> None:
    print("\n\n\n---HANDLE_FOUND_OBJECT CALLED---\n",
          f"  {local_path=}\n  {file_identifier=}\n  {sha256=}\n  {metadata=}\n\n\n")

def handle_modified_object(
    local_path: str,
    file_identifier: str,
    new_sha256: str,
    old_sha256: str,
    metadata: Dict[Hashable, Any],
) -> None:
    print("\n\n\n---HANDLE_MODIFIED_OBJECT CALLED---\n",
          f"  {local_path=}\n  {file_identifier=}\n  {old_sha256=}\n  {new_sha256}\n  {metadata=}\n\n\n")

if __name__ == '__main__':
    num_sample = 10000
    download_dir = "assets/objaverse/alignment_annotation"
    # annotations = oxl.get_annotations(download_dir="assets/objaverse/annotation")
    # annotations
    alignment_annotations = oxl.get_alignment_annotations(download_dir=download_dir)
    # print(alignment_annotations.groupby('source').size())
    # print(alignment_annotations); exit()
    sketchfab_df = alignment_annotations.groupby('source').get_group('sketchfab')
    sampled_df = sketchfab_df.reset_index(drop=True)
    # print(sampled_df); exit()
    oxl.download_objects(
        objects=sampled_df,
        download_dir=os.path.dirname(download_dir),
        processes=4,
        # handle_found_object=handle_found_object,
        # handle_modified_object=handle_modified_object,
        save_repo_format='files',
    )