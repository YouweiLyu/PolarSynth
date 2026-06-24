import os
import time
import uuid
import argparse
import subprocess

import utils.file_util as futil
import utils.sampler_util as sutil
from utils.DataSampler import DataSampler
from utils.MitsubaRenderer import MitsubaRenderer

def main(args):
    #### hardcode part ####
    integrator_type = 'direct'
    camera_dict = {
        'view_distance':15, 'fov':20, 'resolution':(512,512), 
        'sp_type':'stratified', 'sp_num':36, 'rfilter':'mitchell',
    }
    #### ####
    spl_model = DataSampler(args.model_txt_path, args.material_txt_path, args.hdri_txt_path)
    for idx, model_path in enumerate(spl_model.get_model_paths()):
        print(f'#### Rendering [{idx+1:4d}/{spl_model.get_model_num():4d}] scenes ####')
        spl_model.clear_all_cache(args.cache_dir)
        model_name = spl_model.load_model_file(model_path)
        scene_name = model_name

        model_idx_name = f'{scene_name}'
        
        spl_model.init_model_augment_info()
        spl_model.select_material('pplastic', 'assets/materials/forest_ground_04')
        spl_model.cache_material(os.path.join(args.cache_dir, model_name))

        spl_model.select_light_type('envmap', hdri_path='assets/3D_assets/hdri/polyhaven/abandoned_bakery_2k.exr')
        spl_model.init_cam_pose(**camera_dict)

        spls = [spl_model]
        info_save_path = os.path.join(args.cache_dir, 'info.npy')
        sutil.save_multi_sampler_info_for_blender(model_idx_name, info_save_path, spls, spl_model)
        # spl_model.save_info_for_blender(model_idx_name, info_save_path)
        # call blender to generate the shape, albedo, and normal map
        s_time = time.time()
        subprocess.run(['blender', '-b', '-t', f'{args.workers}', '-P', 'blender_postprocess.py',
                            '--', info_save_path, 
                            '--save_dir', args.save_dir, '--cache_dir', args.cache_dir], 
                            check=True, stdout=subprocess.DEVNULL)
        futil.save_blender_results(os.path.join(args.save_dir, model_idx_name), info_save_path, spl_model.get_scene_dup_num())
        e_time = time.time(); print(f'Blender time: {e_time-s_time:.2f}s')
        # update the model path after Blender modification 
        spl_model.update_model_path(os.path.join(args.cache_dir, 'tmp_0.obj'))
        model_info_main = spl_model.get_model_info()
        
        model_infos = [model_info_main]
        camera_infos = spl_model.get_cam_infos()
        light_info = spl_model.get_light_info()
        light_type = light_info['light_type']
        # init mitsuba renderer
        renderer = MitsubaRenderer(args.mi_variant, args.workers)
        renderer.load_integrator(integrator_type)
        renderer.load_model(model_infos)

        render_scene_num = spl_model.get_scene_num()
        render_count = 0
        for scene_idx in range(render_scene_num):
            if render_scene_num > 1:
                print(f'  ## Rendering [{scene_idx+1:4d}/{render_scene_num:4d}] lightings ##')

            renderer.load_light(light_info['envmap'], light_info['otherlights'][scene_idx])

            if light_type in ['envmap', 'envmap_otherlights']:
                cam_infos = camera_infos
            elif light_type in ['otherlights']:
                cam_infos = [camera_infos[scene_idx]]
            else:
                raise ValueError('invalid light type')
            
            for cam_idx, camera_info in enumerate(cam_infos):
                if len(cam_infos) > 1:
                    print(f'  ## Rendering [{cam_idx+1:4d}/{len(cam_infos):4d}] views ##')

                render_count += 1
                save_prefix = os.path.join(args.save_dir, f'{model_idx_name}_{render_count:03d}')
                renderer.load_sensor(camera_info)
                renderer.render(save_prefix)

        if args.debug and idx > 1:
            exit()
            
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
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
    parser.add_argument('--workers', default=8, type=int, help='number of the threads for mitsuba rendering')
    args = parser.parse_args()
    if args.debug:
        print('#### Debug Mode ####')
    id_process = str(uuid.uuid1())
    args.cache_dir = os.path.join(args.cache_dir, id_process)
    print(f'Rendering UUID: {id_process}')
    main(args)
    