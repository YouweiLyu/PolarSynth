import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm

import drjit as dr
import mitsuba as mi
from mitsuba import ScalarTransform4f as T

import sys 
sys.path.append("..") 
import utils.utils as util
import utils.file_util as futil
import utils.render_util as rutil

save_parent_dir = './renderings/polar_scenes_sgl_obj'
scene_file_dir = './scenes'
scene_files = [
    'sphere_materials.xml',
]

#############################################################
#### loading sensor/integrator from xml others from python dict ####
#############################################################

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
    parser.add_argument('--mi_variant', type=str, default='llvm_spectral_polarized', 
                        choices=['scalar_rgb', 'scalar_spectral_polarized', 'llvm_spectral_polarized', 'cuda_spectral_polarized'],
                        help='the variant type of mitsuba renderer')
    parser.add_argument('--model_dir', default=None, 
                        help='Directory to the obj files to be rendered.')
    parser.add_argument('--model_txt_path', default=None, 
                        help='Directory to the obj files to be rendered.')
    parser.add_argument('--material_dir', default=None, 
                        help='Directory to the material files to be rendered.')
    parser.add_argument('--material_txt_path', default=None, 
                        help='Directory to the material files to be rendered.')
    parser.add_argument('--hdri_dir', default=None, 
                        help='Directory to the hdri files to be rendered.')
    parser.add_argument('--hdri_txt_path', default=None, 
                        help='Directory to the hdri files to be rendered.')
    parser.add_argument('--save_dir', type=str, default='renderings/polar_scenes_sgl_obj',
                        help='Directory to save the rendered images.')
    args = parser.parse_args()

    # mitsuba 3.3.0
    mi.set_variant(args.mi_variant)

    # int_kwargs = {'type': 'direct', 'shading_samples': 64}
    int_kwargs = {'type': 'path', 'max_depth': -1, 'rr_depth': 16}
    integrator = rutil.load_integrator(int_kwargs)
    # integrator = mi.load_file('scenes/components/aov_stokes_direct.xml').integrator()
    radius = 30
    phis = [360.0/3 * i for i in range(3)]
    theta = 90
    fov, h, w = 10, 512, 512

    sensors = [
        # rutil.load_sensor(radius, phi, theta, [0,0,0], [-np.cos(np.deg2rad(phi)),-np.sin(np.deg2rad(phi)),0], fov, h, w, 'gaussian') for phi in phis
        rutil.load_sensor(radius, 180, theta, [0,0,0.5], [0,-np.sin(np.deg2rad(phi)),-np.cos(np.deg2rad(phi))], fov, h, w, 'gaussian') for phi in phis
        # mi.load_file('scenes/components/sensor.xml', **{'y':-np.sin(np.deg2rad(phi)),'z':-np.cos(np.deg2rad(phi))}).sensors()[0] for phi in phis
        # mi.load_file('scenes/sphere_materials.xml').sensors()
    ]

    model_path = 'assets/models/sphere.obj'
    model_type = model_path.split('.')[-1].lower()
    model_transform = T.translate([0, 0, 0])
    # lighting
    envmap_int_scale = 400
    envmap_path = 'assets/hdris/9C4A0003-e05009bcad.exr'
    # envmap_dict = rutil.load_envmap(envmap_path, envmap_int_scale, True)
    dist_light_kwargs = {'intensity': 3, 'direction': [1,0,0]}
    dist_light_mat = T.look_at(origin=[2,0,5], target=[0,0,0], up=[0,0,1])
    envmap_dict = rutil.load_distant_light(**dist_light_kwargs)
    dist_light_kwargs['mat'] = np.array(dist_light_mat.matrix)
    uv_transform = T.scale([1, 1, 0])
    albedo_path = 'assets/materials/forest_ground_04/forest_ground_04_4K_Color.jpg'
    albedo = mi.Bitmap(cv2.imread(albedo_path)[...,::-1])
    albedo = mi.Bitmap(np.ones_like(cv2.imread(albedo_path)[...,::-1])*255)
    # roughness = cv2.imread('assets/materials/forest_ground_04/forest_ground_04_4K_Roughness.jpg', -1)/255.
    roughness = 0.01
    normal_var_path = 'assets/materials/forest_ground_04/forest_ground_04_4K_NormalGL.jpg'
    displace_path = 'assets/materials/forest_ground_04/forest_ground_04_4K_Displacement.jpg'
    normal_map = mi.Bitmap(cv2.imread(normal_var_path)[...,::-1])
    displace_map = mi.Bitmap(cv2.imread(displace_path)[...,::-1])
    obj_dict = rutil.load_pplastic_model(albedo, roughness, 
        model_path=model_path, model_type=model_type, uv_transform=uv_transform, dist='ggx',
        normal_variant=None, transform=model_transform)
    # obj_dict = rutil.load_pplastic_model_bump(albedo, roughness, 
    #     model_path=model_path, model_type=model_type, uv_transform=uv_transform, dist='ggx',
    #     displacement=displace_map, transform=model_transform)
    # plane_transform = T.rotate([0,1,0], 90,)
    # obj_dict = rutil.load_plane_pplastic(normal_map, transform=plane_transform)

    scene = mi.load_dict({
        "type": "scene",
        "sensor": {
            "type": "perspective",
            "myfilm": {
                "type": "hdrfilm",
                "width": 16,
                "height": 16,
            }, 
            "mysampler": {
                "type": "independent",
                "sample_count": 4,
            },
        },
        'obj': obj_dict,
        'envmap': envmap_dict,
    })
    
    save_dir = 'renderings/polar_scenes/check_camera_pose_1'
    os.makedirs(save_dir, exist_ok=True)
    for idx, sensor in enumerate(tqdm(sensors)):
        obj_name = os.path.split(model_path)[-1].split('.')[0]
        meta_save_path = os.path.join(save_dir, f'{obj_name}_{idx}_meta.npy')
        rendered_save_path = os.path.join(save_dir, f'{obj_name}_{idx}.exr')
        img_save_path = os.path.join(save_dir, f'{obj_name}_{idx}_img.png')
        albedo_save_path = os.path.join(save_dir, f'{obj_name}_{idx}_albedo.png')
        normal_save_path = os.path.join(save_dir, f'{obj_name}_{idx}_normal.png')
        polarprop_save_path = os.path.join(save_dir, f'{obj_name}_{idx}_pprop.png')
        
        img_rendered = mi.render(scene, sensor=sensor, integrator=integrator)
        world_mat = np.array(mi.traverse(sensor)['to_world'].matrix)[0]
        bitmap = mi.Bitmap(img_rendered, channel_names=['R','G','B']+integrator.aov_names())
        bitmap.write(rendered_save_path)
        futil.save_img_mitsuba_exr(rendered_save_path, img_save_path)
        # futil.save_albedo_mitsuba_exr(rendered_save_path, albedo_save_path)
        futil.save_normal_mitsuba_exr(rendered_save_path, normal_save_path, 65535, world_mat)
        futil.save_polarprop_exr(rendered_save_path, polarprop_save_path)
        
        model_mat = np.array(model_transform.matrix)
        meta_info_dict = {
            'fov': fov,
            'resolutions': (h, w),
            'camera_mat': world_mat,
            'camera_up_axis': 'Z',
            'model_mats': [model_mat],
            'model_paths': [model_path],
            'albedo_paths': [albedo_path],
            # 'roughness_path': 'assets/materials/forest_ground_04/forest_ground_04_4K_Roughness.jpg',
            'normal_paths': [normal_var_path],
            'displace_paths': [displace_path],
            'envmap_path': envmap_path,
            'dist_light_info': [dist_light_kwargs],
        }
        # np.save(meta_save_path, meta_info_dict)
        # print(model_mat); exit()
    # model_paths = util.get_file_paths(args.model_txt_path, args.model_dir)
    # material_paths = util.get_file_paths(args.material_txt_path, args.material_dir)
    # hdri_paths = util.get_file_paths(args.hdri_txt_path, args.hdri_dir)

    # material_num = len(material_paths)
    # hdri_num = len(hdri_paths)
    
    # for m_path in model_paths:
    #     pass
    # for scene_fn in scene_files:
    #     scene_name = scene_fn.split('.')[0]
    #     scene_xml_path = os.path.join(scene_file_dir, scene_fn)
    #     scene = mi.load_file(scene_xml_path)

    #     params = mi.traverse(scene).items()
    #     for k, v in params:
    #         print(k, v)
        
    #     save_dir = os.path.join(save_parent_dir, scene_name)
    #     os.makedirs(save_dir, exist_ok=True)

    #     sensor_count = 4
    #     radius = 40
    #     phis = [360.0/sensor_count * i for i in range(sensor_count)]
    #     theta = 90.0
    #     sensors = [load_sensor(radius, phi, theta) for phi in phis]

    #     for idx, sensor in enumerate(tqdm(sensors)):
    #         rendered_save_path = os.path.join(save_dir, f'{scene_name}_{idx:03d}.exr')
    #         img_save_path = os.path.join(save_dir, f'{scene_name}_{idx:03d}_img.png')
    #         albedo_save_path = os.path.join(save_dir, f'{scene_name}_{idx:03d}_albedo.png')
    #         depth_save_path = os.path.join(save_dir, f'{scene_name}_{idx:03d}_depth.png')
    #         normal_save_path = os.path.join(save_dir, f'{scene_name}_{idx:03d}_normal.png')
    #         polarprop_save_path = os.path.join(save_dir, f'{scene_name}_{idx:03d}_pprop.png')

    #         img_rendered = mi.render(scene, sensor=sensor)
    #         world_mat = np.array(mi.traverse(sensor)['to_world'].matrix)[0]
    #         bitmap = mi.Bitmap(img_rendered, channel_names=['R','G','B']+scene.integrator().aov_names())
            
    #         bitmap.write(rendered_save_path)
    #         futil.save_img_mitsuba_exr(rendered_save_path, img_save_path)
    #         futil.save_albedo_mitsuba_exr(rendered_save_path, albedo_save_path)
    #         futil.save_normal_mitsuba_exr(rendered_save_path, normal_save_path, 65535, world_mat)
    #         futil.save_polarprop_exr(rendered_save_path, polarprop_save_path)