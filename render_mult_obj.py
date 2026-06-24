"""Multi-object polarised renderer.

Drives ``ModelSampler`` (multi-object) + ``SceneSampler`` for ``--num_scene``
scenes. Each scene gets one Blender postprocess pass and a Mitsuba render.
"""
import argparse
import logging
import os
import random

import utils.file_util as futil
import utils.sampler_util as sutil
from utils.cli import add_common_render_args
from utils.DataSampler import ModelSampler, SceneSampler
from utils.logging_util import setup_logging
from utils.MitsubaRenderer import MitsubaRenderer
from utils.pipeline import numbered_prefix, render_views, run_blender

log = logging.getLogger(__name__)


def main(args):
    add_plane_choices = [False, True]
    integrator_type = 'direct'
    camera_dict = {
        'view_distance': 15, 'fov': args.fov, 'resolution': (512, 512),
        'sp_type': 'stratified', 'sp_num': 36, 'rfilter': 'mitchell',
    }

    model_sampler = ModelSampler(
        args.model_txt_path, args.material_txt_path, args.num_obj_per_scene,
        args.cache_dir, args.disable_mesh_disp,
    )
    scene_sampler = SceneSampler(args.hdri_txt_path)
    # Hoist renderer construction out of the per-scene loop (Phase 2 perf win).
    renderer = MitsubaRenderer(args.mi_variant, args.workers, inprocess=getattr(args, "inprocess", False))
    renderer.load_integrator(integrator_type)

    for idx in range(args.num_scene):
        log.info('#### Rendering [%4d/%4d] scenes ####', idx + 1, args.num_scene)
        model_sampler.clear()
        scene_name = model_sampler.sample_meshes(None, random.choice(add_plane_choices))
        model_sampler.augment_mesh_info(args.mesh_superimpose)
        model_sampler.sample_material()
        model_sampler.augment_material()
        model_sampler.cache_material()
        model_info_bl = model_sampler.get_model_info('bl')
        model_glb_info = model_sampler.model_global_info

        scene_sampler.load_model_info(**model_glb_info)
        scene_sampler.sample_light_type()
        scene_sampler.sample_cam_pose(args.pose_num_per_scene, **camera_dict)
        scene_sampler.sample_other_light()
        cam_info_bl = scene_sampler.get_cam_info('bl')

        scene_name = f'{idx+args.start_index:05d}_{args.num_obj_per_scene}{scene_name}'
        info_save_path = os.path.join(args.cache_dir, 'info.npy')
        sutil.save_info_for_blender(scene_name, info_save_path, model_info_bl, cam_info_bl)

        run_blender('blender_postprocess_multobj.py', info_save_path, args.save_dir, args.workers)
        futil.save_blender_results_(
            os.path.join(args.save_dir, scene_name),
            cam_info_bl,
            scene_sampler.get_scene_dup_num(),
        )

        light_info = scene_sampler.get_light_info()
        model_info_mi = model_sampler.get_model_info('mi')
        renderer.load_model(model_info_mi)
        render_views(
            renderer=renderer,
            light_info=light_info,
            cam_infos=scene_sampler.get_cam_info('mi'),
            scene_num=scene_sampler.get_scene_num(),
            save_prefix_fn=numbered_prefix(args.save_dir, scene_name),
        )

        futil.save_summary_info(
            model_sampler.summary_info(),
            scene_sampler.summary_info(),
            os.path.join(args.save_dir, f'{scene_name}.json'),
        )
        if args.debug and idx > 1:
            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Renders multi-object scenes with shared material and light sampling.')
    parser.add_argument('--num_obj_per_scene', default=1, type=int,
                        help='Number of objects per scene.')
    parser.add_argument('--num_scene', default=1, type=int,
                        help='Number of rendered scenes.')
    parser.add_argument('--start_index', default=1, type=int,
                        help='Index offset of the first rendered scene.')
    parser.add_argument('--mesh_superimpose', default=False, action='store_true',
                        help='Allow meshes to overlap in scene composition.')
    parser.add_argument('--disable_mesh_disp', default=True, action='store_false',
                        help='Disable mesh displacement in rendering.')
    parser.add_argument('--fov', default=20, type=int, help='Camera fov in degrees.')
    add_common_render_args(parser)
    args = parser.parse_args()
    setup_logging(debug=args.debug)
    if args.debug:
        log.info('#### Debug Mode ####')
    main(args)
