"""Metal + dielectric hybrid scene renderer.

Each scene is rendered twice: once with the hybrid (metal + plastic) materials,
once with just the dielectric subset. Blender postprocesses both info npy files.
"""
import argparse
import logging
import os

import utils.file_util as futil
import utils.sampler_util as sutil
from utils.cli import add_common_render_args
from utils.DataSampler import ModelSampler, SceneSampler
from utils.logging_util import setup_logging
from utils.MitsubaRenderer import MitsubaRenderer
from utils.pipeline import run_blender, select_cam_infos

log = logging.getLogger(__name__)


def main(args):
    add_plane = False
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
    renderer = MitsubaRenderer(args.mi_variant, args.workers, inprocess=getattr(args, "inprocess", False))
    renderer.load_integrator(integrator_type)

    for idx in range(args.num_scene):
        log.info('#### Rendering [%4d/%4d] scenes ####', idx + 1, args.num_scene)
        model_sampler.clear()
        scene_name = model_sampler.sample_meshes(None, add_plane)
        model_sampler.augment_two_mesh_info()
        model_sampler.sample_2kinds_material()
        model_sampler.augment_material()
        model_sampler.cache_material()
        dielec_model_info_bl = model_sampler.get_material_model_info('pplastic', 'bl')
        hybrid_model_info_bl = model_sampler.get_model_info('bl')
        model_glb_info = model_sampler.model_global_info

        scene_sampler.load_model_info(**model_glb_info)
        scene_sampler.sample_light_type()
        scene_sampler.sample_cam_pose_along_y(args.pose_num_per_scene, **camera_dict)
        scene_sampler.sample_other_light()
        cam_info_bl = scene_sampler.get_cam_info('bl')

        dielec_scene_name = f'{idx+args.start_index:05d}_{args.num_obj_per_scene}{scene_name}_dielec'
        hybrid_scene_name = f'{idx+args.start_index:05d}_{args.num_obj_per_scene}{scene_name}_hybrid'
        dielec_info_save_path = os.path.join(args.cache_dir, 'dielec_info.npy')
        hybrid_info_save_path = os.path.join(args.cache_dir, 'hybrid_info.npy')
        sutil.save_info_for_blender(hybrid_scene_name, hybrid_info_save_path, hybrid_model_info_bl, cam_info_bl)
        sutil.save_info_for_blender(dielec_scene_name, dielec_info_save_path, dielec_model_info_bl, cam_info_bl)

        run_blender('blender_postprocess_multobj.py', hybrid_info_save_path, args.save_dir, args.workers,
                    quiet_stdout=False)
        run_blender('blender_postprocess_multobj.py', dielec_info_save_path, args.save_dir, args.workers,
                    quiet_stdout=False)
        futil.save_blender_results_(
            os.path.join(args.save_dir, hybrid_scene_name), cam_info_bl,
            scene_sampler.get_scene_dup_num(),
        )
        futil.save_blender_results_(
            os.path.join(args.save_dir, dielec_scene_name), cam_info_bl,
            scene_sampler.get_scene_dup_num(),
        )

        light_info = scene_sampler.get_light_info()
        light_type = light_info['light_type']
        dielec_model_info_mi = model_sampler.get_material_model_info('pplastic', 'mi')
        hybrid_model_info_mi = model_sampler.get_model_info('mi')

        # NOTE(phase-2): the loop below preserves a pre-existing quirk where the
        # renderer.load_model(dielec_model_info_mi) inside the inner loop is
        # never reverted, so after the first iteration the file labelled
        # "hybrid_..." actually contains dielec output. Behaviour intentionally
        # kept identical to the pre-refactor version; investigate before fixing.
        renderer.load_model(hybrid_model_info_mi)
        cam_info_mi = scene_sampler.get_cam_info('mi')
        scene_num = scene_sampler.get_scene_num()
        render_count = 0
        for scene_idx in range(scene_num):
            if scene_num > 1:
                log.info('  ## Rendering [%4d/%4d] lightings ##', scene_idx + 1, scene_num)
            renderer.load_light(light_info['envmap'], light_info['otherlights'][scene_idx])
            view_cams = select_cam_infos(light_type, scene_idx, cam_info_mi)
            for cam_idx, camera_info in enumerate(view_cams):
                if len(view_cams) > 1:
                    log.info('  ## Rendering [%4d/%4d] views ##', cam_idx + 1, len(view_cams))
                render_count += 1
                renderer.load_sensor(camera_info)
                renderer.render(os.path.join(args.save_dir, f'{hybrid_scene_name}_{render_count:03d}'))
                renderer.load_model(dielec_model_info_mi)
                renderer.render(os.path.join(args.save_dir, f'{dielec_scene_name}_{render_count:03d}'))

        futil.save_summary_info(
            model_sampler.summary_info(),
            scene_sampler.summary_info(),
            os.path.join(args.save_dir, f'{scene_name}.json'),
        )
        if args.debug and idx > 1:
            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Renders metal + dielectric hybrid scenes, plus a dielectric-only pass.')
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
