"""Shared argparse fragments for the render_*.py entry scripts.

Before Phase 2 every entry script duplicated the same ~30-line argparse block.
``add_common_render_args`` adds the shared flags to any caller-owned parser so
each entry script keeps its own description / extra flags but stops drifting
on the common ones.
"""
from __future__ import annotations

import argparse


MI_VARIANTS = (
    'scalar_rgb',
    'scalar_spectral_polarized',
    'llvm_spectral_polarized',
    'cuda_spectral_polarized',
)


def add_common_render_args(parser: argparse.ArgumentParser) -> None:
    """Register the flags shared across every render_*.py entry point."""
    parser.add_argument('--pose_num_per_scene', default=8, type=int,
                        help='The number of different camera poses in one scene.')
    parser.add_argument('--model_txt_path', default='txts/models_sgl_obj_train.txt',
                        help='Path to the obj-list txt.')
    parser.add_argument('--material_txt_path', default='txts/materials_sgl_obj_train.txt',
                        help='Path to the material-dir-list txt.')
    parser.add_argument('--hdri_txt_path', default='txts/hdri_sgl_obj_train.txt',
                        help='Path to the hdri-list txt.')
    parser.add_argument('--save_dir', type=str, default='renderings/polar_scenes_sgl_obj',
                        help='Directory to save the rendered images.')
    parser.add_argument('--cache_dir', type=str, default='tmp',
                        help='Directory for intermediate cache files.')
    parser.add_argument('--mi_variant', type=str, default='llvm_spectral_polarized',
                        choices=MI_VARIANTS,
                        help='Mitsuba variant to use.')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Enable debug mode (verbose logs + early break).')
    parser.add_argument('--workers', default=8, type=int,
                        help='Threads passed to mitsuba/blender.')
    parser.add_argument('--start_model_index', default=0, type=int,
                        help='Skip the first N model indices.')
    parser.add_argument('--inprocess', default=False, action='store_true',
                        help='Render Mitsuba in-process (mi.render) instead of '
                             'shelling out to the CLI. Faster but needs the '
                             'mitsuba Python package available at import time.')
