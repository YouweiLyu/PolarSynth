"""Blender-side preprocessing for the multi-object pipeline (with optional
metal/dielec split for the hybrid renderer).

Example:
    blender -b -P blender_postprocess_multobj.py -- info.npy --save_dir <dir>

Driven by ``render_mult_obj.py`` and ``render_metal_dielec_obj.py``.
"""
import argparse
import os
import sys

import bpy
import numpy as np

from blender_postprocess_common import (  # noqa: E402
    DEFAULT_ALBEDO_PATH, DEFAULT_ROUGH_PATH,
    apply_displacement, apply_smart_uv_and_augment,
    build_camera, clear_scene_objects_and_images,
    import_and_smooth_mesh, init_render, render_camera_pass,
    split_into_material_collections, wire_principled_material,
)


# The multi-object schema applies an extra X-axis flip when seating each mesh
# in world space (see git history: ``ac3a8ae``). We replicate it here exactly.
_MULTI_OBJ_AXES_FLIP = np.asarray([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def _parse_args():
    parser = argparse.ArgumentParser(description='Bake multi-obj scenes for Mitsuba.')
    parser.add_argument('meta_info_path', type=str, default='./tmp/info.npy')
    parser.add_argument('--save_dir', type=str, default='renderings/polar_scenes_sgl_obj')
    parser.add_argument('--color_depth', type=str, default='32')
    parser.add_argument('--format', type=str, default='OPEN_EXR_MULTILAYER')
    parser.add_argument('--engine', type=str, default='CYCLES')
    argv = sys.argv[sys.argv.index("--") + 1:]
    return parser.parse_args(argv)


def main():
    args = _parse_args()
    context, scene, render, extra_layers = init_render(
        args, extra_view_layers=('dielec', 'metal'),
    )
    view_layer_dielec, view_layer_metal = extra_layers
    if render.engine == 'CYCLES':
        scene.cycles.samples = 1

    clear_scene_objects_and_images()

    meta_info = np.load(args.meta_info_path, allow_pickle=True).item()
    os.makedirs(args.save_dir, exist_ok=True)
    save_name = meta_info['save_name']
    cam_infos = meta_info['camera_infos']
    model_infos = meta_info['model_infos']

    metal_obj_names, dielec_obj_names = [], []
    for model_idx, model_info in enumerate(model_infos):
        model_name = model_info['mesh_name']
        model_path = model_info['mesh_path']
        if model_path.split('.')[-1].lower() != 'obj':
            raise ValueError('3D model file is not a .obj file!')

        cache_dir = model_info['cache_dir']
        smooth_level = model_info['smooth_level']
        smooth_cache_path = os.path.join(cache_dir, f'{model_name}_smooth{smooth_level:d}.obj')

        obj_name = f"{model_name}_{model_idx}"
        # NOTE: pre-Phase-4 behaviour: the multi-object script applies an
        # extra X-axis flip when seating world-mat. Kept verbatim.
        world_mat = np.asarray(model_info['transform']) @ _MULTI_OBJ_AXES_FLIP
        obj = import_and_smooth_mesh(
            obj_path=model_path,
            obj_name=obj_name,
            world_mat=world_mat,
            smooth_level=smooth_level,
            smooth_level_scale=model_info['smooth_level_scale'],
            vertex_num=model_info['vertex_num'],
            smooth_cache_path=smooth_cache_path,
        )

        brdf = model_info['brdf']
        if 'conductor' in brdf:
            metal_obj_names.append(obj_name)
        elif 'pplastic' in brdf:
            dielec_obj_names.append(obj_name)

        apply_smart_uv_and_augment(
            obj,
            uv_transform=model_info['uv_transform'],
            uv_transform_center=model_info['uv_transform_center'],
        )
        apply_displacement(
            obj, disp_path=model_info['disp_path'],
            strength=model_info['disp_strength'], model_idx=model_idx,
        )

        # Persist per-model OBJ for Mitsuba.
        model_save_path = os.path.join(cache_dir, 'tmp.obj')
        bpy.ops.export_scene.obj(
            filepath=model_save_path, use_selection=True,
            use_normals=False, use_materials=False, use_uvs=True,
            axis_forward='Y', axis_up='Z',
        )

        albedo_path = model_info['albedo_path'] or DEFAULT_ALBEDO_PATH
        rough_path = model_info['rough_path'] or DEFAULT_ROUGH_PATH
        wire_principled_material(
            obj, albedo_path=albedo_path, rough_path=rough_path,
            model_idx=model_idx, albedo_color_space='Non-Color',
        )

    # Split into per-material collections so the holdout view layers can
    # render the metal / dielec masks.
    split_into_material_collections(
        scene, dielec_obj_names, metal_obj_names,
        view_layer_dielec=view_layer_dielec,
        view_layer_metal=view_layer_metal,
    )

    cam = build_camera(scene)
    for idx, cam_info in enumerate(cam_infos):
        render_camera_pass(
            cam=cam, scene=scene, render=render,
            cam_world_mat=cam_info['transform'],
            up_axis=cam_info['up_axis'],
            fov=cam_info['fov'],
            resolution=cam_info['resolution'],
            save_path=os.path.join(args.save_dir, f"{save_name}_{idx+1:03d}_bl.exr"),
        )


if __name__ == '__main__':
    main()
