"""Blender-side preprocessing for the single-object pipeline.

Example:
    blender -b -P blender_postprocess.py -- info.npy --save_dir <dir>

Driven by ``render_sgl_obj.py``: imports each mesh listed in the info npy,
runs the smoothing pyramid (subdiv vs remesh), applies displacement +
albedo + roughness, renders one frame per camera transform.
"""
import argparse
import os
import sys

import bpy
import numpy as np

# Local sibling module — Blender runs us with cwd at the repo root.
from blender_postprocess_common import (  # noqa: E402
    DEFAULT_ALBEDO_PATH, DEFAULT_ROUGH_PATH,
    apply_displacement, apply_smart_uv_and_augment,
    build_camera, clear_scene_objects_and_images,
    import_and_smooth_mesh, init_render,
    render_camera_pass, wire_principled_material,
)


def _parse_args():
    parser = argparse.ArgumentParser(description='Bake Blender mesh + materials for Mitsuba.')
    parser.add_argument('meta_info_path', type=str, default='./tmp/info.npy')
    parser.add_argument('--cache_dir', type=str, default='./tmp')
    parser.add_argument('--save_dir', type=str, default='renderings/polar_scenes_sgl_obj')
    parser.add_argument('--color_depth', type=str, default='32')
    parser.add_argument('--format', type=str, default='OPEN_EXR_MULTILAYER')
    parser.add_argument('--engine', type=str, default='CYCLES')
    argv = sys.argv[sys.argv.index("--") + 1:]
    return parser.parse_args(argv)


def main():
    args = _parse_args()
    context, scene, render, _ = init_render(args)
    clear_scene_objects_and_images()

    meta_info = np.load(args.meta_info_path, allow_pickle=True).item()
    os.makedirs(args.save_dir, exist_ok=True)
    save_name = meta_info['save_name']

    for model_idx in range(len(meta_info['model_paths'])):
        model_name = meta_info['model_names'][model_idx]
        model_path = meta_info['model_paths'][model_idx]
        if model_path.split('.')[-1].lower() != 'obj':
            raise ValueError('3D model file is not a .obj file!')

        model_cache_dir = os.path.join(meta_info['model_cache_dirs'][model_idx], '.cached')
        os.makedirs(model_cache_dir, exist_ok=True)
        smooth_level = meta_info['model_smooth_levels'][model_idx]
        smooth_cache_path = os.path.join(model_cache_dir, f'{model_name}_smooth{smooth_level:d}.obj')
        if os.path.isfile(smooth_cache_path):
            # Skip the slow smoothing pass; load the cached smoothed mesh.
            model_path = smooth_cache_path

        obj_name = f"{os.path.basename(model_path).split('.')[0]}_{model_idx}"
        obj = import_and_smooth_mesh(
            obj_path=model_path,
            obj_name=obj_name,
            world_mat=meta_info['model_transforms'][model_idx],
            smooth_level=smooth_level,
            smooth_level_scale=meta_info['model_smooth_level_scales'][model_idx],
            vertex_num=meta_info['model_vertex_nums'][model_idx],
            smooth_cache_path=smooth_cache_path,
        )

        apply_smart_uv_and_augment(
            obj,
            uv_transform=meta_info['uv_transforms'][model_idx],
            uv_transform_center=meta_info['uv_transform_centers'][model_idx],
        )
        apply_displacement(
            obj,
            disp_path=meta_info['displacement_paths'][model_idx],
            strength=meta_info['model_disp_strengths'][model_idx],
            model_idx=model_idx,
        )

        # Persist the geometry post-modification for Mitsuba to consume.
        model_save_path = os.path.join(args.cache_dir, f'tmp_{model_idx}.obj')
        bpy.ops.export_scene.obj(
            filepath=model_save_path, use_selection=True,
            use_normals=False, use_materials=False, use_uvs=True,
            axis_forward='Y', axis_up='Z',
        )

        albedo_path = meta_info['albedo_diff_paths'][model_idx] or DEFAULT_ALBEDO_PATH
        rough_path = meta_info['roughness_paths'][model_idx] or DEFAULT_ROUGH_PATH
        wire_principled_material(
            obj, albedo_path=albedo_path, rough_path=rough_path,
            model_idx=model_idx,
        )

    # Render every camera.
    cam = build_camera(scene)
    for idx, cam_mat in enumerate(meta_info['camera_transforms']):
        render_camera_pass(
            cam=cam, scene=scene, render=render,
            cam_world_mat=cam_mat,
            up_axis=meta_info['camera_up_axes'][idx],
            fov=meta_info['camera_fovs'][idx],
            resolution=meta_info['camera_resolutions'][idx],
            save_path=os.path.join(args.save_dir, f"{save_name}_{idx+1:03d}_bl.exr"),
        )


if __name__ == '__main__':
    main()
