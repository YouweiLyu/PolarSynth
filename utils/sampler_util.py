import os
import numpy as np

def load_path_text(path):
    if os.path.isfile(path):
        with open(path, 'r') as f:
            paths = f.read().splitlines()
        return paths
    else:
        print(f'cannot find the txt file: {path}, using the default setting')
        return []

def save_multi_sampler_info_for_blender(save_name, info_save_path, model_samplers, cam_sampler):
    info = {
        'save_name': save_name,
        # model
        'model_cache_dirs': [spl.cache_dir for spl in model_samplers],
        'model_names': [spl.model_info['name'] for spl in model_samplers],
        'model_paths': [spl.model_info['path_origin'] for spl in model_samplers],
        'model_centers': [spl.model_info['bbox_center'] for spl in model_samplers],
        'model_transforms': [np.array(spl.model_info['transform'].matrix) for spl in model_samplers],
        'model_disp_strengths': [spl.model_info['disp_strength'] for spl in model_samplers],
        'model_smooth_levels': [spl.model_info['smooth_level'] for spl in model_samplers],
        'model_smooth_level_scales': [spl.model_info['smooth_level_scale'] for spl in model_samplers],
        'model_vertex_nums': [spl.model_info['vertex_num'] for spl in model_samplers],
        'uv_transforms': [spl.model_info['uv_transform'] for spl in model_samplers],
        'uv_transform_centers': [spl.model_info['uv_transform_center'] for spl in model_samplers],
        # material
        'displacement_paths': [spl.material_info['displacement_path'] for spl in model_samplers],
        'albedo_diff_paths': [spl.material_info['albedo_diff_path'] for spl in model_samplers],
        'roughness_paths': [spl.material_info['roughness_path'] for spl in model_samplers],
        # camera
        'camera_fovs': cam_sampler.cam_info['fovs'],
        'camera_resolutions': cam_sampler.cam_info['resolutions'],
        'camera_transforms': [np.array(item.matrix) for item in cam_sampler.cam_info['transforms']],
        'camera_up_axes': cam_sampler.cam_other_info['up_axes'],
    }
    np.save(info_save_path, info)

def save_info_for_blender(save_name, info_save_path, model_infos, camera_infos):
    info = {
        'save_name': save_name,
        'model_infos': model_infos,
        'camera_infos': camera_infos,
    }
    np.save(info_save_path, info)