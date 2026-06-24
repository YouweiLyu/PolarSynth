import os
import argparse
import numpy as np
from tqdm import tqdm

import mitsuba as mi
from mitsuba import ScalarTransform4f as T

import sys 
sys.path.append("..") 
import utils.file_util as futil
import utils.render_util as rutil

#############################################################
#### loading scenes from xml directly ###
#############################################################

save_parent_dir = './renderings/polar_scenes'
scene_file_dir = './scenes'
scene_files = [
    'sphere_materials_1.xml',
    'sphere_materials_2.xml',
    'sphere_materials_3.xml',
]

if __name__ == '__main__':
    # mitsuba 3.3.0
    mi.set_variant('llvm_spectral_polarized')

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', default=False, action='store_true')
    args = parser.parse_args()

    for idx, scene_fn in enumerate(scene_files):
        scene_name = scene_fn.split('.')[0]
        scene_xml_path = os.path.join(scene_file_dir, scene_fn)
        scene = mi.load_file(scene_xml_path)

        save_dir = os.path.join(save_parent_dir, scene_name)
        os.makedirs(save_dir, exist_ok=True)

        ####
        # idx = 9
        # rendered_save_path = os.path.join(save_dir, f'{scene_name}_{idx:03d}.exr')
        # img_save_path = os.path.join(save_dir, f'{scene_name}_{idx:03d}_img.png')
        # albedo_save_path = os.path.join(save_dir, f'{scene_name}_{idx:03d}_albedo.png')
        # depth_save_path = os.path.join(save_dir, f'{scene_name}_{idx:03d}_depth.png')
        # normal_save_path = os.path.join(save_dir, f'{scene_name}_{idx:03d}_normal.png')
        # polarprop_save_path = os.path.join(save_dir, f'{scene_name}_{idx:03d}_pprop.png')

        # img_rendered = mi.render(scene)
        # # world_mat = np.array(mi.traverse(sensor)['to_world'].matrix)[0]
        # bitmap = mi.Bitmap(img_rendered, channel_names=['R','G','B']+scene.integrator().aov_names())
        
        # bitmap.write(rendered_save_path)
        # futil.save_img_mitsuba_exr(rendered_save_path, img_save_path)
        # futil.save_albedo_mitsuba_exr(rendered_save_path, albedo_save_path)
        # futil.save_normal_mitsuba_exr(rendered_save_path, normal_save_path)
        # futil.save_polarprop_exr(rendered_save_path, polarprop_save_path)
        ####

        # sensor_count = 3
        # radius = 40
        # phis = [360.0/sensor_count * i for i in range(sensor_count)]
        # theta = 90.0
        # sensors = [rutil.load_sensor(radius, 0, theta, up=[0,-np.sin(np.deg2rad(phi)),np.cos(np.deg2rad(phi))], fov=30, h=512, w=512) for phi in phis]
        # sensors = [mi.load_file('scenes/components/sensor.xml', **{'y':-np.sin(np.deg2rad(phi)),'z':-np.cos(np.deg2rad(phi))}).sensors()[0] for phi in phis]

        # for idx, sensor in enumerate(tqdm(sensors)):
        rendered_save_path = os.path.join(save_dir, f'{scene_name}_{idx:03d}.exr')
        img_save_path = os.path.join(save_dir, f'{scene_name}_{idx:03d}_img.png')
        albedo_save_path = os.path.join(save_dir, f'{scene_name}_{idx:03d}_albedo.png')
        depth_save_path = os.path.join(save_dir, f'{scene_name}_{idx:03d}_depth.png')
        normal_save_path = os.path.join(save_dir, f'{scene_name}_{idx:03d}_normal.png')
        polarprop_save_path = os.path.join(save_dir, f'{scene_name}_{idx:03d}_pprop.png')

        img_rendered = mi.render(scene)
        world_mat = np.array(mi.traverse(scene.sensors()[0])['to_world'].matrix)[0]
        bitmap = mi.Bitmap(img_rendered, channel_names=['R','G','B']+scene.integrator().aov_names())
        
        bitmap.write(rendered_save_path)
        futil.save_img_mitsuba_exr(rendered_save_path, img_save_path)
        futil.save_albedo_mitsuba_exr(rendered_save_path, albedo_save_path)
        futil.save_normal_mitsuba_exr(rendered_save_path, normal_save_path, 65535, world_mat)
        futil.save_polarprop_exr(rendered_save_path, polarprop_save_path)