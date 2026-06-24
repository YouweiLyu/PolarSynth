"""Single-object polarised renderer.

Per object (from --model_txt_path), generates --num_rendered_per_obj augmented
variants and renders each across multiple lighting + camera samples.

Usage:
    python render_sgl_obj.py --model_txt_path txts/... [--num_rendered_per_obj N] ...
"""
import argparse
import logging
import os
import random

import utils.file_util as futil
import utils.sampler_util as sutil
from utils.cli import add_common_render_args
from utils.DataSampler import DataSampler
from utils.logging_util import setup_logging
from utils.MitsubaRenderer import MitsubaRenderer
from utils.pipeline import numbered_prefix, render_views, run_blender

log = logging.getLogger(__name__)


def main(args):
    add_plane_choices = [False, False]
    plane_model_path = 'assets/models/default/plane.obj'
    integrator_type = 'direct'
    camera_dict = {
        'view_distance': 15, 'fov': 20, 'resolution': (512, 512),
        'sp_type': 'stratified', 'sp_num': 36, 'rfilter': 'mitchell',
    }

    spl_model = DataSampler(args.model_txt_path, args.material_txt_path, args.hdri_txt_path)
    spl_plane = DataSampler('', args.material_txt_path, '')
    # Hoist renderer construction out of the per-variant loop (Phase 2 perf win).
    renderer = MitsubaRenderer(args.mi_variant, args.workers, inprocess=getattr(args, "inprocess", False))
    renderer.load_integrator(integrator_type)

    for idx, model_path in enumerate(spl_model.get_model_paths()):
        log.info('#### Rendering [%4d/%4d] scenes ####', idx + 1, spl_model.get_model_num())
        spl_model.clear_all_cache(args.cache_dir)
        spl_plane.clear_all_cache(args.cache_dir)
        model_name = spl_model.load_model_file(model_path)
        scene_name = model_name
        plane_name = spl_plane.load_model_file(plane_model_path)

        for model_idx in range(args.start_model_index, args.num_rendered_per_obj):
            add_plane = random.choice(add_plane_choices)
            log.info(' ### Rendering [%4d/%4d] model variants of [%4d/%4d] scenes ###',
                     model_idx + 1, args.num_rendered_per_obj,
                     idx + 1, spl_model.get_model_num())
            model_idx_name = f'{scene_name}_{model_idx+1:03d}'

            spl_model.sample_model_augment_info()
            spl_model.sample_material()
            spl_model.augment_material()
            spl_model.cache_material(os.path.join(args.cache_dir, model_name))

            spl_plane.sample_plane_augment_info()
            spl_plane.sample_material()
            spl_plane.augment_material()
            spl_plane.cache_material(os.path.join(args.cache_dir, plane_name))

            spl_model.sample_light_type()
            spl_model.sample_cam_pose(args.pose_num_per_scene, **camera_dict)
            spl_model.sample_other_light()

            spls = [spl_model, spl_plane] if add_plane else [spl_model]
            info_save_path = os.path.join(args.cache_dir, 'info.npy')
            sutil.save_multi_sampler_info_for_blender(model_idx_name, info_save_path, spls, spl_model)

            run_blender(
                'blender_postprocess.py',
                info_save_path,
                args.save_dir,
                args.workers,
                extra_args=['--cache_dir', args.cache_dir],
            )
            futil.save_blender_results(
                os.path.join(args.save_dir, model_idx_name),
                info_save_path,
                spl_model.get_scene_dup_num(),
            )

            # update the model path after Blender modification
            spl_model.update_model_path(os.path.join(args.cache_dir, 'tmp_0.obj'))
            model_info_main = spl_model.get_model_info()
            if add_plane:
                spl_plane.update_model_path(os.path.join(args.cache_dir, 'tmp_1.obj'))
                model_info_plane = spl_plane.get_model_info()
                model_infos = [model_info_main, model_info_plane]
            else:
                model_infos = [model_info_main]

            camera_infos = spl_model.get_cam_infos()
            light_info = spl_model.get_light_info()
            renderer.load_model(model_infos)

            render_views(
                renderer=renderer,
                light_info=light_info,
                cam_infos=camera_infos,
                scene_num=spl_model.get_scene_num(),
                save_prefix_fn=numbered_prefix(args.save_dir, model_idx_name),
            )

        if args.debug and idx > 1:
            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Renders a single object across material / lighting / view samples.')
    parser.add_argument('--num_rendered_per_obj', default=1, type=int,
                        help='The number of rendered images of a single object.')
    add_common_render_args(parser)
    args = parser.parse_args()
    setup_logging(debug=args.debug)
    if args.debug:
        log.info('#### Debug Mode ####')
    main(args)
