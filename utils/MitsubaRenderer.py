"""Mitsuba 3 scene-dict wrapper.

Two rendering paths exist:

* :meth:`render` — original CLI subprocess path. Writes XML, shells out to
  ``mitsuba`` for each view. Compatible with the LLVM/CUDA backend selection
  used historically.
* :meth:`render_inprocess` — Phase 3 addition. Builds the scene once via
  ``mi.load_dict`` and renders with ``mi.render``. Avoids per-view process
  spawn + JIT warmup. Opt-in via ``MitsubaRenderer(inprocess=True)`` or by
  calling the method directly.
"""
from __future__ import annotations

import logging
import os
import time

import mitsuba as mi
from mitsuba import ScalarTransform4f as T

import utils.file_util as futil
import utils.render_util as rutil
from utils.process_utils import run

log = logging.getLogger(__name__)


class MitsubaRenderer():
    def __init__(self, mi_variant='llvm_spectral_polarized', workers=8, inprocess=False):
        self.set_defaults(mi_variant, workers)
        self.inprocess = inprocess

    def set_defaults(self, mi_variant, workers):
        mi.set_variant(mi_variant)
        self.mi_variant = mi_variant
        self.scene_dict = {'type': 'scene'}
        self.workers = workers

    def load_integrator(self, int_type='direct'):
        assert int_type in ['direct', 'path'], f'Invalid integrator type: {int_type}'
        int_kwargs_dict = {
            'direct': {'type': 'direct', 'shading_samples': 32},
            'path': {'type': 'path', 'max_depth': -1, 'rr_depth': 16},
        }
        int_dict = {
            'type': 'stokes',
            'int': int_kwargs_dict[int_type],
        }
        self.aov_names = mi.load_dict(int_dict).aov_names()
        self.scene_dict['integrator'] = int_dict

    def _resolve_material_paths(self, model_info):
        """Pull (albedo, roughness, model_path) out of either schema variant."""
        if 'albedo_diff' in model_info:
            albedo = model_info['albedo_diff']
        else:
            albedo = model_info['albedo_path']
        if 'roughness' in model_info:
            roughness = model_info['roughness']
        else:
            roughness = model_info['rough_path']
        if 'path' in model_info:
            model_path = model_info['path']
        elif 'cache_dir' in model_info:
            model_path = os.path.join(model_info['cache_dir'], 'tmp.obj')
        else:
            raise ValueError(
                'model_info has neither "path" nor "cache_dir"; cannot locate mesh.'
            )
        return albedo, roughness, model_path

    def load_model(self, model_infos):
        self.clear_model()
        for idx, model_info in enumerate(model_infos):
            brdf = model_info['brdf']
            albedo, roughness, model_path = self._resolve_material_paths(model_info)
            model_type = model_path.split('.')[-1]
            model_transform = T.translate([0, 0, 0])
            if brdf in ['pplastic', 'roughpplastic']:
                model_dict = rutil.load_pplastic_model(
                    albedo, roughness,
                    ior=model_info['ior'], model_path=model_path,
                    model_type=model_type, dist='ggx', transform=model_transform,
                )
            elif brdf == 'conductor':
                model_dict = rutil.load_conductor_model(
                    model_info['conductor'], model_path, model_type, transform=model_transform,
                )
            elif brdf == 'roughconductor':
                model_dict = rutil.load_roughconductor_model(
                    model_info['conductor'], roughness, model_path,
                    dist='ggx', model_type=model_type, transform=model_transform,
                )
            elif brdf == 'dielectric':
                model_dict = rutil.load_dielectric_model(
                    model_info['ior'],
                    model_path=model_path, model_type=model_type, transform=model_transform,
                )
            else:
                raise NotImplementedError(f'Unsupported BRDF: {brdf!r}')

            self.scene_dict[f'model_{idx}'] = model_dict

    def load_sensor(self, cam_info_kwargs):
        self.scene_dict['sensor'] = rutil.load_sensor_(**cam_info_kwargs)

    def load_light(self, env_info, otherlight_info):
        self.clear_light()
        if env_info:
            if env_info['hdri_path']:
                envmap_dict = rutil.load_envmap(env_info['hdri_path'], env_info['hdri_scale'], True)
            else:
                envmap_dict = rutil.load_const_env(env_info['hdri_scale'])
            self.scene_dict['envmap'] = envmap_dict
        if otherlight_info:
            for idx, lt_dict in enumerate(otherlight_info):
                self.scene_dict[f'otherlight_{idx+1}'] = lt_dict

    def clear_light(self):
        for key in list(self.scene_dict):
            if 'envmap' in key or 'otherlight' in key:
                self.scene_dict.pop(key, None)

    def clear_model(self):
        for key in list(self.scene_dict):
            if 'model' in key:
                self.scene_dict.pop(key, None)

    def clear_sensor(self):
        self.scene_dict.pop('sensor', None)

    def render(self, save_prefix):
        """Dispatch to inprocess vs subprocess based on the constructor flag."""
        if self.inprocess:
            return self.render_inprocess(save_prefix)
        return self.render_subprocess(save_prefix)

    def render_subprocess(self, save_prefix):
        """Original behaviour: write XML, shell out to the ``mitsuba`` CLI."""
        xml_path = f'{save_prefix}.xml'
        mi.xml.dict_to_xml(self.scene_dict, xml_path, False, False)
        t0 = time.time()
        run(['mitsuba', '-m', self.mi_variant, '-t', f'{self.workers}',
             '-o', f'{save_prefix}_mi.exr', xml_path])
        log.info('Mitsuba %s time: %.2fs', self.mi_variant, time.time() - t0)
        chan_names = ['R', 'G', 'B'] + self.aov_names
        futil.save_mitsuba_results(save_prefix, chan_names)

    def render_inprocess(self, save_prefix):
        """Phase 3 addition: render in-process, skipping process spawn + XML round-trip.

        ``mi.load_dict(self.scene_dict)`` builds a ``mi.Scene`` once per call;
        we then ``mi.render(scene)`` and save the bitmap. The result file is
        the same EXR layout the subprocess path produces, so all downstream
        ``futil.save_*_mitsuba_exr`` helpers continue to work unchanged.
        """
        t0 = time.time()
        scene = mi.load_dict(self.scene_dict)
        img = mi.render(scene)
        chan_names = ['R', 'G', 'B'] + self.aov_names
        out_path = f'{save_prefix}_mi.exr'
        mi.Bitmap(img, channel_names=chan_names).write(out_path)
        log.info('Mitsuba %s inprocess time: %.2fs', self.mi_variant, time.time() - t0)
        # Re-use the same post-process helper to derive the JPG / PNG products.
        futil.save_mitsuba_results(save_prefix, chan_names)
