"""Case-study renderer.

Sweeps a deterministic grid (materials x albedos x roughnesses x envmaps) over
a fixed mesh. Used for qualitative comparisons rather than data generation.
"""
import argparse
import logging
import os

import utils.file_util as futil
import utils.sampler_util as sutil
from utils.cli import MI_VARIANTS
from utils.DataSamplerCaseStudy import DataSampler
from utils.logging_util import setup_logging
from utils.MitsubaRenderer import MitsubaRenderer
from utils.pipeline import render_views, run_blender

log = logging.getLogger(__name__)


def _save_prefix_factory(save_dir, scene_name):
    # Pre-refactor behaviour: every render uses the same prefix and overwrites.
    # Kept intentionally to match prior output naming.
    def _fn(_idx):
        return os.path.join(save_dir, scene_name)
    return _fn


def main(args):
    integrator_type = 'direct'
    camera_dict = {
        'view_distance': 10, 'fov': 20, 'resolution': (512, 512),
        'sp_type': 'stratified', 'sp_num': 36, 'rfilter': 'mitchell',
    }
    models = ['assets/models/default/sphere.obj']
    materials = [
        'conductor:Au',
        'roughconductor:Cu',
        'pplastic',
    ]
    albedos = [
        'assets/materials/default/white.png',
        'assets/materials/default/red.png',
        'assets/materials/default/yellow.png',
    ]
    roughs = [0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    # Envmaps inherit MITSUBA_HDRI_DIR via env-var defaults; this list can be
    # overridden via a future config file (Phase 5).
    from utils import paths as default_paths
    envmaps = [
        os.path.join(default_paths.HDRI_DIR, 'polyhaven', f)
        for f in ('abandoned_bakery_2k.exr', 'art_studio_2k.exr', 'autumn_hockey_2k.exr')
    ]

    spl_model = DataSampler('', '', '')
    renderer = MitsubaRenderer(args.mi_variant, args.workers, inprocess=getattr(args, "inprocess", False))
    renderer.load_integrator(integrator_type)

    for idx, model_path in enumerate(models):
        log.info('#### Rendering [%4d/%4d] scenes ####', idx + 1, len(models))
        spl_model.clear_all_cache(args.cache_dir)
        model_name = spl_model.load_model_file(model_path)
        total_render_num = len(materials) * len(albedos) * len(roughs) * len(envmaps)
        render_num = 0
        for material in materials:
            m_name = "".join(material.split(':'))
            kind = material.split(':')[0]
            if kind == 'conductor':
                albedo_paths, rough_values = [None], [None]
            elif kind == 'roughconductor':
                albedo_paths, rough_values = [None], roughs
            elif kind == 'pplastic':
                albedo_paths, rough_values = albedos, roughs
            else:
                raise ValueError(f'Unsupported material kind: {kind!r}')

            for albedo_path in albedo_paths:
                a_name = (
                    os.path.basename(albedo_path).split('.')[0].split('_')[0].lower()
                    if albedo_path is not None else ''
                )
                for rough_value in rough_values:
                    r_name = f'r{rough_value:.3f}' if rough_value is not None else ''
                    for envmap_path in envmaps:
                        env_name = os.path.basename(envmap_path).split('_')[0]
                        scene_name = '_'.join(c for c in (m_name, a_name, r_name, env_name) if c)
                        log.info(' ### Rendering [%4d/%4d] variants of [%4d/%4d] scenes ###',
                                 render_num + 1, total_render_num, idx + 1, len(models))

                        spl_model.sample_model_augment_info()
                        spl_model.assign_material(material, albedo_path, rough_value)
                        spl_model.cache_material(os.path.join(args.cache_dir, model_name))

                        spl_model.assign_light_type('envmap', envmap_path)
                        spl_model.assign_cam_pose(args.pose_num_per_scene, **camera_dict)

                        info_save_path = os.path.join(args.cache_dir, 'info.npy')
                        sutil.save_multi_sampler_info_for_blender(
                            scene_name, info_save_path, [spl_model], spl_model,
                        )
                        run_blender(
                            'blender_postprocess.py',
                            info_save_path,
                            args.save_dir,
                            args.workers,
                            extra_args=['--cache_dir', args.cache_dir],
                        )
                        futil.save_blender_results(
                            os.path.join(args.save_dir, scene_name),
                            info_save_path,
                            spl_model.get_scene_dup_num(),
                        )

                        spl_model.update_model_path(os.path.join(args.cache_dir, 'tmp_0.obj'))
                        renderer.load_model([spl_model.get_model_info()])

                        render_views(
                            renderer=renderer,
                            light_info=spl_model.get_light_info(),
                            cam_infos=spl_model.get_cam_infos(),
                            scene_num=spl_model.get_scene_num(),
                            save_prefix_fn=_save_prefix_factory(args.save_dir, scene_name),
                        )

                        render_num += 1
                        if args.debug and idx > 1:
                            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Renders a parameter sweep over materials, albedos, roughness and HDRIs.')
    parser.add_argument('--pose_num_per_scene', default=8, type=int,
                        help='Number of camera poses per scene.')
    parser.add_argument('--save_dir', type=str,
                        default='renderings/polar_scenes_sgl_obj',
                        help='Directory to save the rendered images.')
    parser.add_argument('--cache_dir', type=str, default='tmp',
                        help='Directory for intermediate cache files.')
    parser.add_argument('--mi_variant', type=str, default='llvm_spectral_polarized',
                        choices=MI_VARIANTS, help='Mitsuba variant.')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Enable debug mode.')
    parser.add_argument('--workers', default=8, type=int,
                        help='Threads for mitsuba rendering.')
    parser.add_argument('--inprocess', default=False, action='store_true',
                        help='Render Mitsuba in-process (mi.render) instead of '
                             'shelling out to the CLI.')
    args = parser.parse_args()
    setup_logging(debug=args.debug)
    if args.debug:
        log.info('#### Debug Mode ####')
    main(args)
