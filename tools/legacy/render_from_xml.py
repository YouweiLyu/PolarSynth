import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

import OpenEXR as openexr
import mitsuba as mi
from mitsuba import ScalarTransform4f as T

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import utils.file_util as futil
import utils.render_util as rutil

save_parent_dir = './renderings/polar_scenes'
scene_file_dir = './scenes'
scene_files = [
    # 'sphere_materials_test_lights.xml',
    # 'real_env_dragon.xml',
    # 'dragon_normal.xml',
    'real_env_dragon_wo_uv.xml',
    'real_env_dragon_uv0.05.xml',
    'real_env_dragon_uv20.xml',
]

if __name__ == '__main__':
    # mitsuba 3.3.0
    mi.set_variant('llvm_spectral_polarized')

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', default=False, action='store_true')
    args = parser.parse_args()

    for scene_fn in scene_files:
        scene_name = scene_fn.split('.')[0]
        scene_xml_path = os.path.join(scene_file_dir, scene_fn)
        scene = mi.load_file(scene_xml_path)

        # params = mi.traverse(scene).items()
        # for k, v in params:
        #     print(k, v)

        save_dir = os.path.join(save_parent_dir, scene_name)
        os.makedirs(save_dir, exist_ok=True)

        sensor_count = 2
        radius = 40
        phis = [360.0/sensor_count * i for i in range(sensor_count)]
        theta = 90.0
        sensors = [rutil.load_sensor(radius, phi, theta, up=[0,0,1], fov=30, h=512, w=512) for phi in phis]

        for idx, sensor in enumerate(tqdm(sensors)):
            assert isinstance(sensor, mi.Sensor)
            
            rendered_save_path = os.path.join(save_dir, f'{scene_name}_{idx:03d}.exr')
            img_save_path = os.path.join(save_dir, f'{scene_name}_{idx:03d}_img.png')
            albedo_save_path = os.path.join(save_dir, f'{scene_name}_{idx:03d}_albedo.png')
            depth_save_path = os.path.join(save_dir, f'{scene_name}_{idx:03d}_depth.png')
            normal_save_path = os.path.join(save_dir, f'{scene_name}_{idx:03d}_normal.png')
            polarprop_save_path = os.path.join(save_dir, f'{scene_name}_{idx:03d}_pprop.png')

            img_rendered = mi.render(scene, sensor=sensor)
            world_mat = np.array(mi.traverse(sensor)['to_world'].matrix)[0]
            bitmap = mi.Bitmap(img_rendered, channel_names=['R','G','B']+scene.integrator().aov_names())
            
            bitmap.write(rendered_save_path)
            print(rendered_save_path)
            exr_file = openexr.InputFile(rendered_save_path)
            futil.save_img_mitsuba_exr(exr_file, img_save_path)
            # futil.save_albedo_mitsuba_exr(exr_file, albedo_save_path)
            futil.save_normal_mitsuba_exr(exr_file, normal_save_path, 65535, world_mat)
            # futil.save_polarprop_exr(rendered_save_path, polarprop_save_path)