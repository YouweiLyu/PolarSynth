####################################
# python renderer_sgl_obj.py 
####################################
import os
import argparse
import subprocess

import utils.file_util as futil
import utils.sampler_util as sutil
from utils.DataSampler import DataSampler
from utils.MitsubaRenderer import MitsubaRenderer

def main(args):
    #### hardcode part ####
    plane_model_path = 'assets/models/default/plane.obj'
    integrator_type = 'direct'
    camera_dict = {
        'view_distance':15, 'fov':20, 'resolution':(512,512), 
        'sp_type':'stratified', 'sp_num':36, 'rfilter':'mitchell',
    }
    #### #### #### #### ####
    # spl_model = DataSampler(args.model_txt_path, args.material_txt_path, args.hdri_txt_path)
    spl_plane = DataSampler('', args.material_txt_path, '')
    idx = 0
    print(f'#### Rendering [{idx+1:4d}/{1:4d}] scenes ####')
    # model_path = next(spl_model.get_model_paths())
    # model_name = spl_model.load_model_file(model_path)
    scene_name = 'Plane'

    model_idx = 0
    print(f' ### Rendering [{model_idx+1:4d}/{1:4d}] model variants of [{idx+1:4d}/{1:4d}] scene###')
    futil.clear_folder_files(args.cache_dir)
    scene_idx_name = f'{scene_name}_{model_idx+1:03d}'
    
    # spl_model.sample_model_augment_info()
    # # spl_model.sample_material()
    # spl_model.select_material('pplastic', material_dir='assets/materials/ambientcg/Tape/Tape001')
    # spl_model.augment_material()
    # spl_model.cache_material(args.cache_dir)

    spl_plane.load_model_file(plane_model_path)
    spl_plane.sample_plane_augment_info()
    # spl_plane.sample_material()
    spl_plane.select_material('pplastic', material_dir='assets/3D_assets/materials/ambientcg/AcousticFoam/AcousticFoam001')
    spl_plane.augment_material()
    spl_plane.cache_material(os.path.join(args.cache_dir, f'model_{0:02d}'))

    spl_plane.select_light_type('envmap')
    # spl_plane.sample_light_type()
    spl_plane.sample_cam_pose(args.pose_num_per_scene, **camera_dict)
    spl_plane.sample_other_light()
    
    # spl_model.sample_light_type()
    # spl_model.sample_cam_pose(args.pose_num_per_scene, **camera_dict)
    # spl_model.sample_other_light()
    info_save_path = os.path.join(args.cache_dir, 'info.npy')
    # spl_model.save_info_for_blender(scene_idx_name, info_save_path)
    # sutil.save_multi_sampler_info_for_blender(scene_idx_name, info_save_path, [spl_model, spl_plane], spl_model)
    sutil.save_multi_sampler_info_for_blender(scene_idx_name, info_save_path, [spl_plane], spl_plane)
    # call blender to generate the shape, albedo, and normal map
    subprocess.run(['blender', '-b', '-P', 'blender_postprocess.py',
                        '--', info_save_path, 
                        '--save_dir', args.save_dir, '--cache_dir', args.cache_dir], 
                        check=True, stdout=subprocess.DEVNULL, )
    # futil.save_blender_results(os.path.join(args.save_dir, scene_idx_name), info_save_path, spl_model.get_scene_dup_num())
    futil.save_blender_results(os.path.join(args.save_dir, scene_idx_name), info_save_path, spl_plane.get_scene_dup_num())

    # update the model path after Blender modification 
    # spl_model.update_model_path(os.path.join(args.cache_dir, 'tmp_0.obj'))
    spl_plane.update_model_path(os.path.join(args.cache_dir, f'tmp_0.obj'))
    
    # get rendering info from the sampler
    # model_info_main = spl_model.get_model_info()
    model_info_plane = spl_plane.get_model_info()
    # camera_infos = spl_model.get_cam_infos()
    # light_info = spl_model.get_light_info()
    camera_infos = spl_plane.get_cam_infos()
    light_info = spl_plane.get_light_info()
    light_type = light_info['light_type']
    # init mitsuba renderer
    renderer = MitsubaRenderer()
    renderer.load_integrator(integrator_type)
    # renderer.load_model([model_info_main, model_info_plane])
    renderer.load_model([model_info_plane])

    render_scene_num = 1
    render_count = 0
    scene_idx = 0
    if render_scene_num > 1:
        print(f'  ## Rendering [{scene_idx+1:4d}/{render_scene_num:4d}] lightings ##')

    renderer.load_light(light_info['envmap'], light_info['otherlights'][scene_idx])

    if light_type in ['envmap', 'envmap_otherlights']:
        cam_infos = camera_infos
    elif light_type in ['otherlights']:
        cam_infos = [camera_infos[scene_idx]]
    else:
        raise ValueError('invalid light type')
    
    cam_idx = 0
    camera_info = cam_infos[cam_idx]
    if len(cam_infos) > 1:
        print(f'  ## Rendering [{cam_idx+1:4d}/{len(cam_infos):4d}] views ##')

    render_count += 1
    save_prefix = os.path.join(args.save_dir, f'{scene_idx_name}_{render_count:03d}')
    renderer.load_sensor(camera_info)
    renderer.render(save_prefix)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
    parser.add_argument('--num_rendered_per_obj', default=1, type=int,
                        help='The number of rendered images of a single object.')
    parser.add_argument('--pose_num_per_scene', default=8, type=int,
                        help='The number of different camera poses in one scene.')
    parser.add_argument('--model_txt_path', default='txts/models_sgl_obj_train.txt', 
                        help='Directory to the obj files to be rendered.')
    parser.add_argument('--material_txt_path', default='txts/materials_sgl_obj_train.txt', 
                        help='Directory to the material files to be rendered.')
    parser.add_argument('--hdri_txt_path', default='txts/hdri_sgl_obj_train.txt', 
                        help='Directory to the hdri files to be rendered.')
    parser.add_argument('--save_dir', type=str, default='renderings/polar_scenes_sgl_obj',
                        help='Directory to save the rendered images.')
    parser.add_argument('--cache_dir', type=str, default='tmp',
                        help='Directory to save the intermediate results.')
    parser.add_argument('--mi_variant', type=str, default='llvm_spectral_polarized', 
                        choices=['scalar_rgb', 'scalar_spectral_polarized', 'llvm_spectral_polarized', 'cuda_spectral_polarized'],
                        help='the variant type of mitsuba renderer')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='enable the debug mode')
    args = parser.parse_args()
    main(args)
    