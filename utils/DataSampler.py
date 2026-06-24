import cv2
import copy
import logging
import numpy as np
from mitsuba import ScalarTransform4f as T
import os
import random
import shutil

from . import utils as util
from . import render_util as rutil
from . import sampler_util as sutil
from . import file_util as futil
from . import paths as default_paths

log = logging.getLogger(__name__)

class DataSampler():
    def __init__(self, model_txt_path, material_txt_path, hdri_txt_path):
        self.set_default(model_txt_path, material_txt_path, hdri_txt_path)
        
    def set_default_params(self):
        self.brdf = None
        self.model_info = {}
        self.material_info = {}
        self.light_info = {}
        self.cam_info = {}
        self.cam_other_info = {}
        self.cache_dir = ''

    def clear_except_model(self):
        self.brdf = None
        self.material_info.clear()
        self.light_info.clear()
        self.cam_info.clear()
        self.cam_other_info.clear()
        self.cache_dir = ''

    def clear_model(self):
        self.model_info.clear()

    def set_default(self, model_txt_path, material_txt_path, hdri_txt_path):
        self.set_default_params()
        model_paths = sutil.load_path_text(model_txt_path)
        self.model_type = '.obj'
        self.model_num = len(model_paths)
        self.model_paths = iter(model_paths if model_paths else ['sphere'])
        self.model_disps = (True, True, False)
        self.model_smooth_levels = (0, 0, 1, 2)
        
        self.brdfs = ('conductor', 'roughconductor', 'pplastic', 'roughpplastic')
        self.material_paths = sutil.load_path_text(material_txt_path)
        self.hdri_paths = sutil.load_path_text(hdri_txt_path)
        
        self.conductor_materials = (
            'a-C', 'Ag', 'Al', 'AlAs', 'Au', 'Be', 'Cr', 'Cu', 'Cu2O', 
            'Hg', 'HgTe', 'Ir', 'K', 'Li', 'MgO', 'Mo', 'Na_palik', 'Nb', 'Ni_palik', 'Rh', 
            'Se', 'SiC', 'Ta', 'Te', 'TiC', 'TiN', 'TiO2', 'VC', 'V_palik', 'VN', 'W',
        )
        self.lights = ('envmap', 'envmap_otherlights', 'otherlights')
        self.smooth_mtl_lights = ('envmap', 'envmap_otherlights')
        self.otherlight_types = ('point', 'spot', 'directional', 'projector')
        self.otherlight_nums = (1, 2, 3)
        
        self.cam_pose_sampling_methods = ('fibonacci_spiral', 'handcrafted')
        self.scene_dup_num = 1

    def get_model_paths(self):
        return self.model_paths
    
    def get_model_num(self):
        return self.model_num
    
    def load_model_file(self, model_path):
        model_name = util.to_camel_case(os.path.basename(model_path).split('.')[0])
        if not model_path.endswith(self.model_type):
            raise ValueError('Unsupported model type!')
        log.debug('model_path=%r', model_path)
        model_info_path = model_path.replace(self.model_type, '_objinfo.npy')
        if not os.path.isfile(model_info_path):
            raise FileNotFoundError(f'model info file invalid: {model_info_path}')
        self.model_info = np.load(model_info_path, allow_pickle=True).item()
        self.model_info['path_origin'] = model_path
        self.model_info['name'] = model_name
        return model_name
    
    def update_model_path(self, new_path):
        if not os.path.isfile(new_path):
            raise FileNotFoundError(f'updated model file invalid: {new_path}')
        self.model_info['path'] = new_path
        self.model_info['type'] = new_path.split('.')[-1].lower()
        
    def init_model_augment_info(self):
        self.model_info['is_disp'] = False
        self.model_info['disp_strength'] = 0
        self.model_info['smooth_level'] = 0
        self.model_info['smooth_level_scale'] = 0
        self.model_info['transform'] = T.translate([0, 0, 0])
        self.model_info['uv_transform'] = (1, 1)
        self.model_info['uv_transform_center'] = (0.5, 0.5)
    
    def sample_model_augment_info(self):
        self.model_info['is_disp'] = bool((not self.model_info['is_disp_prohibit'])*random.choice(self.model_disps))
        self.model_info['disp_strength'] = self.model_info['is_disp']*random.uniform(0.01, 0.2)
        self.model_info['smooth_level'] = random.choice(self.model_smooth_levels)
        self.model_info['smooth_level_scale'] = self.model_info['smooth_level']*random.uniform(0.65, 0.95)
        self.model_info['transform'] = T.translate([0, 0, 0])
        uv_transform = 1. / np.random.uniform(1.25, 20)
        uv_transform = (uv_transform, uv_transform)
        uv_transform_center = (np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9))
        log.debug('uv_transform_center=%r', uv_transform_center)
        log.debug('uv_transform=%r', uv_transform)
        self.model_info['uv_transform'] = uv_transform
        self.model_info['uv_transform_center'] = uv_transform_center

    def sample_plane_augment_info(self):
        scale_factor = np.random.uniform(1.0, 2)
        self.model_info['is_disp'] = random.choice([True, False])
        self.model_info['disp_strength'] = self.model_info['is_disp']*random.uniform(0.1, 0.3)
        self.model_info['smooth_level'] = 0
        self.model_info['smooth_level_scale'] = self.model_info['smooth_level']*random.uniform(0.65, 0.95)
        self.model_info['transform'] = T.scale([scale_factor, scale_factor, 1])
        uv_transform = np.random.uniform(0.8, 1.0)
        uv_transform = (uv_transform, uv_transform)
        uv_transform_center = (np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9))
        log.debug('uv_transform_center=%r', uv_transform_center)
        log.debug('uv_transform=%r', uv_transform)
        self.model_info['uv_transform'] = uv_transform
        self.model_info['uv_transform_center'] = uv_transform_center
    
    def _sample_material_dir(self):
        assert self.model_info
        attempt_counter = 0
        attempt_max = 200
        if self.material_paths:
            while(True):
                material_split_path = random.choice(self.material_paths)
                material_dirs = util.list_dir(material_split_path)
                if material_dirs:
                    material_dir = random.choice(material_dirs)
                    if self.model_info.get('is_disp', False):
                        m_name = os.path.split(material_dir)[-1]
                        disp_path = os.path.join(material_dir, f'{m_name}_4K_Displacement.jpg')
                        if os.path.isfile(disp_path):
                            break
                        else:
                            attempt_counter += 1
                            if attempt_counter >= attempt_max:
                                raise FileNotFoundError('Disp is required but not provided in material dir')
                            continue
                    else:
                        break
                else:
                    raise FileNotFoundError(f'Material directory not found in {material_split_path}')
        else:
            material_dir = ''
            self.model_info['is_disp'] = False
            print('Using the default material setting. '
                  'Disp is disabled')
        log.debug('material_dir=%r', material_dir)
        return material_dir

    def sample_material(self):
        assert self.model_info
        self.material_info.clear()
        self.brdf = random.choice(self.brdfs)
        log.debug('self.brdf=%r', self.brdf)
        if 'pplastic' in self.brdf or 'dielectric' in self.brdf:
            model_ior = np.random.uniform(1.30, 1.70)
            conductor_material = None
        elif 'conductor' in self.brdf:
            model_ior = None
            conductor_material = random.choice(self.conductor_materials)
        else:
            raise ValueError(f'Unsupported BRDF: {self.brdf!r}')
        self.material_info['ior'] = model_ior
        self.material_info['conductor'] = conductor_material
        material_dir = self._sample_material_dir()
        self.material_info['material_dir'] = material_dir
        disp, albedo, rough = util.load_material(material_dir)
        if not self.model_info.get('is_disp', False):
            disp = None
        if self.brdf in ['roughconductor', 'conductor', 'dielectric']:
            albedo = None
        if self.brdf in ['conductor', 'dielectric']:
            rough = None
        self.material_info['displacement'] = disp
        self.material_info['albedo_diff'] = albedo
        self.material_info['roughness'] = rough

    def select_material(self, brdf, material_dir=None):
        if material_dir is None:
            material_dir = default_paths.DEFAULT_MATERIAL_DIR
        assert self.model_info
        self.material_info.clear()
        if brdf not in self.brdfs:
            raise ValueError(f'Invalid brdf type: {brdf}')
        self.brdf = brdf
        log.debug('self.brdf=%r', self.brdf)
        if 'pplastic' in self.brdf or 'dielectric' in self.brdf:
            model_ior = np.random.uniform(1.30, 1.70)
            conductor_material = None
        elif 'conductor' in self.brdf:
            model_ior = None
            conductor_material = random.choice(self.conductor_materials)
        else:
            raise ValueError(f'Unsupported BRDF: {self.brdf!r}')
        self.material_info['ior'] = model_ior
        self.material_info['conductor'] = conductor_material
        self.material_info['material_dir'] = material_dir
        disp, albedo, rough = util.load_material(material_dir)
        if any(item is None for item in [disp, albedo, rough]):
            raise ValueError(f'{material_dir} not contains all material!')
        if not self.model_info.get('is_disp', False):
            disp = None
        if self.brdf in ['roughconductor', 'conductor', 'dielectric']:
            albedo = None
        if self.brdf in ['conductor', 'dielectric']:
            rough = None
        self.material_info['displacement'] = disp
        self.material_info['albedo_diff'] = albedo
        self.material_info['roughness'] = rough
    
    def augment_material(self):
        disp = self.material_info['displacement']
        albedo = self.material_info['albedo_diff']
        roughness = self.material_info['roughness']

        rot_angle = np.random.randint(0, 360)
        rot_center = (np.random.uniform(0.2,0.8), np.random.uniform(0.2,0.8))
        log.debug('rot_angle=%r', rot_angle)
        log.debug('rot_center=%r', rot_center)
        disp = rutil.rotate_image(disp, rot_angle, rot_center)
        albedo = rutil.rotate_image(albedo, rot_angle, rot_center)
        roughness = rutil.rotate_image(roughness, rot_angle, rot_center)

        albedo, albedo_hsv_augment = rutil.color_augmentor(albedo)
        roughness, roughness_mean = rutil.roughness_augmentor(roughness, self.brdf)

        disp = rutil.to_sgl_chan_img(disp)
        roughness = rutil.to_sgl_chan_img(roughness)
        
        self.material_info['displacement'] = disp
        self.material_info['albedo_diff'] = albedo
        self.material_info['roughness'] = roughness
        self.material_info['material_augment'] = {
            'rot_angle': rot_angle,
            'rot_center': rot_center,
            'albedo_hsv': albedo_hsv_augment,
            'roughness_mean':  roughness_mean,
        }
    
    def cache_material(self, cache_dir='./tmp'):
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir
        self.clear_cache()
        img_names = ['displacement', 'albedo_diff', 'roughness']
        for img_name in img_names:
            if self.material_info[img_name] is not None:
                img = self.material_info[img_name]
                save_path = os.path.join(cache_dir, f'{img_name}.png')
                cv2.imwrite(save_path, img)
                self.material_info[f'{img_name}_path'] = os.path.join(cache_dir, f'{img_name}.png')
            else:
                self.material_info[f'{img_name}_path'] = None

    def sample_light_type(self):
        assert self.brdf
        if self.brdf in ['conductor', 'dielectric']:
            light_type = random.choice(self.smooth_mtl_lights)
        else:
            light_type = random.choice(self.lights)
        log.debug('light_type=%r', light_type)
        if light_type == 'envmap':
            if self.hdri_paths:
                hdri_path = random.choice(self.hdri_paths)
            else:
                print('Empty hdri path, using the constant env illumination!')
                hdri_path = ''
            env_int_scale = np.random.uniform(0.50,0.70) # hardcoded
            other_lt_type, other_lt_num = None, 0
            self.scene_dup_num = 1
        elif light_type == 'envmap_otherlights':
            if self.hdri_paths:
                hdri_path = random.choice(self.hdri_paths)
            else:
                print('Empty hdri path, using the constant env illumination!')
                hdri_path = ''
            env_int_scale = np.random.uniform(0.90,1.10) # hardcoded
            other_lt_type = random.choice(self.otherlight_types)
            other_lt_num = random.choice(self.otherlight_nums)
            log.debug('hdri_path=%r', hdri_path)
            log.debug('other_lt_type=%r', other_lt_type)
            log.debug('other_lt_num=%r', other_lt_num)
            self.scene_dup_num = 1
        elif light_type == 'otherlights':
            other_lt_type = random.choice(self.otherlight_types)
            other_lt_num = 1
            hdri_path, env_int_scale = None, None
            log.debug('other_lt_type=%r', other_lt_type)
            log.debug('other_lt_num=%r', other_lt_num)
            self.scene_dup_num = other_lt_num
        else:
            raise ValueError(f'Invalid light type "{light_type}"')
        
        self.light_info = {
            'light_type': light_type,
            'hdri_path': hdri_path,
            'hdri_scale': env_int_scale,
            'otherlight_type': other_lt_type,
            'otherlight_num': other_lt_num,
        }
    
    def select_light_type(
            self,
            light_type,
            other_lt_type=None,
            other_lt_num=0,
            hdri_path=None):
        if hdri_path is None:
            hdri_path = default_paths.DEFAULT_HDRI_PATH
        if light_type not in self.lights:
            raise ValueError(f'Invalid light type: {light_type}')
        log.debug('light_type=%r', light_type)
        if light_type == 'envmap':
            if not os.path.isfile(hdri_path):
                raise FileNotFoundError(f'HDRI file {hdri_path} not found')
            log.debug('hdri_path=%r', hdri_path)
            env_int_scale = np.random.uniform(0.50,0.70) # hardcoded
            other_lt_type, other_lt_num = None, 0
        elif light_type == 'envmap_otherlights':
            if other_lt_type not in self.otherlight_types:
                raise ValueError(f'Invalid otherlight type: {light_type}')
            env_int_scale = np.random.uniform(0.90,1.10) # hardcoded
            log.debug('hdri_path=%r', hdri_path)
            log.debug('other_lt_type=%r', other_lt_type)
            log.debug('other_lt_num=%r', other_lt_num)
        elif light_type == 'otherlights':
            if other_lt_type not in self.otherlight_types:
                raise ValueError(f'Invalid otherlight type: {light_type}')
            other_lt_num = 1
            hdri_path, env_int_scale = None, None
            log.debug('other_lt_type=%r', other_lt_type)
            log.debug('other_lt_num=%r', other_lt_num)
            self.scene_dup_num = other_lt_num
        else:
            raise ValueError(f'Invalid light type "{light_type}"')
        
        self.light_info = {
            'light_type': light_type,
            'hdri_path': hdri_path,
            'hdri_scale': env_int_scale,
            'otherlight_type': other_lt_type,
            'otherlight_num': other_lt_num,
        }

    def sample_other_light(self):
        assert self.model_info
        assert self.light_info
        assert self.cam_info
        light_type = self.light_info['light_type']
        if 'otherlights' in light_type:
            other_lt_num = self.light_info['otherlight_num']
            other_lt_type = self.light_info['otherlight_type']
            view_num = len(self.cam_info['fovs'])
            model_center = self.model_info['bbox_center']
            if light_type == 'envmap_otherlights':
                scene_other_lt_dirs = [rutil.sample_envmap_otherlight_dirs(other_lt_num)]
            elif light_type == 'otherlights':
                cam_view_dirs_dup = rutil.duplicate_view_dirs(self.cam_other_info['view_dirs'], other_lt_num)
                scene_other_lt_dirs = []
                for i in range(view_num):
                    rutil.sample_otherlight_dirs(scene_other_lt_dirs, other_lt_num, cam_view_dirs_dup[i], True)
            else:
                raise NotImplementedError()
            
            if other_lt_type == 'point':
                other_lt_infos = rutil.get_point_light_info(scene_other_lt_dirs, model_center, light_type)
            elif other_lt_type == 'spot':
                other_lt_infos = rutil.get_spot_light_info(scene_other_lt_dirs, model_center, light_type)
            elif other_lt_type == 'projector':
                other_lt_infos = rutil.get_projector_light_info(scene_other_lt_dirs, model_center, light_type)
            elif other_lt_type == 'directional':
                other_lt_infos = rutil.get_directional_light_info(scene_other_lt_dirs, light_type)
            else:
                raise TypeError(f'invalid type of other lights: "{other_lt_type}"')
            
            self.light_info['otherlights'] = other_lt_infos
        else:
            pass

    def init_cam_pose(self, view_distance=15, fov=20, resolution=(512,512), sp_type='stratified', sp_num=81, rfilter='mitchell'):
        assert self.model_info
        assert self.light_info
        cam_view_dirs = rutil.cube_sampling()
        model_center = self.model_info['bbox_center']
        cam_positions = model_center[None] + view_distance * cam_view_dirs
        cam_world_mats = []
        cam_resolutions = []
        cam_up_axes = []
        cam_fovs = []
        sp_types = []
        sp_nums = []
        rfilters = []
        for i in range(cam_positions.shape[0]):
            cam_position = cam_positions[i]
            cam_world_mats.append(T.look_at(
                origin=cam_position,
                target=model_center,
                up=[0, 0, 1]
            ))
            cam_up_axes.append('Z')
            cam_fovs.append(fov)
            cam_resolutions.append(resolution)
            sp_types.append(sp_type)
            sp_nums.append(sp_num)
            rfilters.append(rfilter)
        
        self.cam_info['fovs'] = cam_fovs
        self.cam_info['resolutions'] = cam_resolutions
        self.cam_info['transforms'] = cam_world_mats.copy()
        self.cam_info['sp_types'] = sp_types
        self.cam_info['sp_nums'] = sp_nums
        self.cam_info['rfilters'] = rfilters

        self.cam_other_info['view_dirs'] = cam_view_dirs.copy()
        self.cam_other_info['up_axes'] = copy.deepcopy(cam_up_axes)


    def sample_cam_pose(self, pose_num, view_distance=15, fov=20, resolution=(512,512), sp_type='stratified', sp_num=81, rfilter='mitchell'):
        assert self.model_info
        assert self.light_info
        if self.model_info['is_symmetric']:
            sample_num = max(min(pose_num//4, 4), 1)
            if self.light_info['light_type'] == 'otherlights':
                sample_num = 1
        else:
            sample_num = pose_num
        
        cam_pose_sampling = random.choice(self.cam_pose_sampling_methods)
        if self.model_info['is_flat']:
            cam_view_dirs = rutil.flat_obj_sampling(sample_num)
        elif cam_pose_sampling == 'fibonacci_spiral':
            cam_view_dirs = rutil.hemisphere_spiral_sampling(sample_num)
        elif cam_pose_sampling == 'handcrafted':
            cam_view_dirs = rutil.handcrafted_sampling(sample_num)
        else:
            raise ValueError(f'Unsupported cam_pose_sampling method: {cam_pose_sampling!r}')

        model_center = self.model_info['bbox_center']
        cam_positions = model_center[None] + view_distance * cam_view_dirs
        cam_world_mats = []
        cam_resolutions = []
        cam_up_axes = []
        cam_fovs = []
        sp_types = []
        sp_nums = []
        rfilters = []
        for i in range(cam_positions.shape[0]):
            cam_position = cam_positions[i]
            cam_world_mats.append(T.look_at(
                origin=cam_position,
                target=model_center,
                up=[0, 0, 1]
            ))
            cam_up_axes.append('Z')
            cam_fovs.append(fov)
            cam_resolutions.append(resolution)
            sp_types.append(sp_type)
            sp_nums.append(sp_num)
            rfilters.append(rfilter)
        
        self.cam_info['fovs'] = cam_fovs
        self.cam_info['resolutions'] = cam_resolutions
        self.cam_info['transforms'] = cam_world_mats.copy()
        self.cam_info['sp_types'] = sp_types
        self.cam_info['sp_nums'] = sp_nums
        self.cam_info['rfilters'] = rfilters

        self.cam_other_info['view_dirs'] = cam_view_dirs.copy()
        self.cam_other_info['up_axes'] = copy.deepcopy(cam_up_axes)

    def save_info_for_blender(self, save_name, info_save_path):
        info = {
            'save_name': save_name,
            # model
            'cache_dir': self.cache_dir,
            'model_names': [self.model_info['name']],
            'model_paths': [self.model_info['path_origin']],
            'model_centers': [self.model_info['bbox_center']],
            'model_transforms': [np.array(self.model_info['transform'].matrix)],
            'model_disp_strengths': [self.model_info['disp_strength']],
            'model_smooth_levels': [self.model_info['smooth_level']],
            'model_smooth_level_scales': [self.model_info['smooth_level_scale']],
            'model_vertex_nums': [self.model_info['vertex_num']],
            'uv_transforms': [self.model_info['uv_transform']],
            'uv_transform_centers': [self.model_info['uv_transform_center']],
            # material
            'displacement_paths': [self.material_info['displacement_path']],
            'albedo_diff_paths': [self.material_info['albedo_diff_path']],
            'roughness_paths': [self.material_info['roughness_path']],
            # camera
            'camera_fovs': self.cam_info['fovs'],
            'camera_resolutions': self.cam_info['resolutions'],
            'camera_transforms': [np.array(item.matrix) for item in self.cam_info['transforms']],
            'camera_up_axes': self.cam_other_info['up_axes'],
        }
        np.save(info_save_path, info)

    def get_model_info(self):
        model_dict = {
            'brdf': self.brdf,
            'name': self.model_info['name'],
            'type': self.model_info['type'],
            'path': self.model_info['path'],
            'transform': T.translate([0, 0, 0]),
            'albedo_diff': self.material_info['albedo_diff_path'],
            'roughness': self.material_info['roughness_path'],
            'ior': self.material_info['ior'],
            'conductor': self.material_info['conductor'],
        }
        return model_dict
    
    def get_cam_infos(self):
        cam_view_num = len(self.cam_info['fovs'])
        cam_infos = []
        for i in range(cam_view_num):
            cam_infos.append({key[:-1]: self.cam_info[key][i] for key in self.cam_info})
        return cam_infos
    
    def get_light_info(self):
        light_type = self.light_info['light_type']
        light_info = {
            'light_type': light_type,
            'envmap': None,
            'otherlights': [None]*self.get_scene_num(),
        }
        if 'envmap' in light_type:
            light_info['envmap'] = {
                'hdri_path': self.light_info['hdri_path'],
                'hdri_scale': self.light_info['hdri_scale'],
            }
        if 'otherlights' in light_type:
            light_info['otherlights'] = self.light_info['otherlights']

        return light_info

    def get_scene_num(self):
        if self.light_info['light_type'] == 'otherlights':
            return len(self.cam_info['fovs']) * self.light_info['otherlight_num']
        else:
            return 1
        
    def get_scene_dup_num(self):
        return self.scene_dup_num
    
    def clear_cache(self):
        if os.path.isdir(self.cache_dir):
            futil.clear_files(self.cache_dir)

    def clear_all_cache(self, cache_dir):
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir)

class ModelInfo():
    def __init__(self):
        self.brdf = None
        self.mesh_name = ''
        self.mesh_path = ''
        self.is_disp = False
        self.is_disp_prohibit = False
        self.transform = T.translate([0, 0, 0])
        self.disp_strength = 0
        self.smooth_level = 0
        self.smooth_level_scale = 0
        self.vertex_num = None
        self.uv_transform = None 
        self.uv_transform_center = (0.5, 0.5)
        self.disp_path = None
        self.albedo_path = None
        self.rough_path = None
        self.disp = None
        self.albedo = None
        self.rough = None
        self.ior = 1.4
        self.material_augment = None
        self.cache_dir = 'tmp'

    def update_params(self, **kwargs):
        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, val)

    def clear_cache(self):
        if os.path.isdir(self.cache_dir):
            futil.clear_files(self.cache_dir)

class ModelSampler():
    """Sampler for mesh models in the scene
    Args: 
        mesh_txt_path: filepath of the txt consisting of mesh model paths
        material_txt_path: filepath of the txt consisting of mesh model paths
        model_num: the number of sampled mesh models
        is_add_plane: add a plane to the scene or not. default: False
    """
    def __init__(self, mesh_txt_path, material_txt_path, model_num, cache_dir='tmp', is_mesh_disp=True):
        self.set_default(mesh_txt_path, material_txt_path, model_num, cache_dir, is_mesh_disp)

    def set_default(self, mesh_txt_path, material_txt_path, model_num, cache_dir, is_mesh_disp):
        self.cache_dir = cache_dir
        self.model_num = model_num
        mesh_paths = sutil.load_path_text(mesh_txt_path)
        self.mesh_paths = mesh_paths if mesh_paths else ['sphere']
        self.mesh_type = '.obj'
        if is_mesh_disp:
            self.mesh_disps = (True, True, False)
        else:
            self.mesh_disps = [False]
        self.mesh_smooth_levels = (0, 0, 1, 2)
        self.model_global_info = {
            'model_center': [0,0,0], 
            'is_model_symmetric': False, 
            'is_model_flat': False,
        }
        
        self.brdfs_ext = ('conductor', 'roughconductor', 'pplastic', 'roughpplastic', 'pplastic')
        self.brdfs = ('conductor', 'roughconductor', 'pplastic', 'roughpplastic')
        self.material_paths = sutil.load_path_text(material_txt_path)
        self.conductor_materials = (
            'a-C', 'Ag', 'Al', 'AlAs', 'Au', 'Be', 'Cr', 'Cu', 'Cu2O', 
            'Hg', 'HgTe', 'Ir', 'K', 'Li', 'MgO', 'Mo', 'Na_palik', 'Nb', 'Ni_palik', 'Rh', 
            'Se', 'SiC', 'Ta', 'Te', 'TiC', 'TiN', 'TiO2', 'VC', 'V_palik', 'VN', 'W',
        )
        self.lights = ('envmap', 'envmap_otherlights', 'otherlights')
        self.smooth_mtl_lights = ('envmap', 'envmap_otherlights')
        self.otherlight_types = ('point', 'spot', 'directional', 'projector')
        self.otherlight_nums = (1, 2, 3)
        
        self.cam_pose_sampling_methods = ('fibonacci_spiral', 'handcrafted')
        self.scene_dup_num = 1        
    
    def sample_meshes(self, mesh_paths_assigned=None, is_add_plane=False):
        model_num = self.model_num
        self.model_info = [ModelInfo() for _ in range(model_num)]
        if mesh_paths_assigned is None:
            mesh_paths = random.choices(self.mesh_paths, k=model_num)
        else:
            if len(mesh_paths_assigned) > model_num:
                mesh_paths = mesh_paths_assigned[:model_num]
            else:
                mesh_paths = (mesh_paths_assigned * np.ceil(len(mesh_paths_assigned)/model_num))[:model_num]
        
        if is_add_plane:
            self.plane_info = ModelInfo()
        else:
            self.plane_info = None
        
        mesh_names = []
        global_centers = []
        global_symmetric, global_flat = True, True
        for idx, mesh_path in enumerate(mesh_paths):
            if not mesh_path.endswith(self.mesh_type):
                raise ValueError(f'Unsupported model type: {mesh_path.split(".")[-1]}')
            mesh_info_path = mesh_path.replace(self.mesh_type, '_objinfo.npy')
            if not os.path.isfile(mesh_info_path):
                raise FileNotFoundError(f'Model info file invalid: {mesh_info_path}')
            mesh_info = np.load(mesh_info_path, allow_pickle=True).item()
            mesh_info['mesh_path'] = mesh_path
            mesh_name = util.to_camel_case(os.path.basename(mesh_path).split('.')[0])
            mesh_info['mesh_name'] = mesh_name
            mesh_names.append(mesh_name[:3])
            global_symmetric *= mesh_info['is_symmetric']
            global_flat *= mesh_info['is_flat']
            global_centers.append(mesh_info['bbox_center'])
            self.model_info[idx].update_params(**mesh_info)

        self.model_global_info['is_model_symmetric'] = global_symmetric
        self.model_global_info['is_model_flat'] = global_flat
        model_center = np.stack(global_centers, 0).mean(axis=0)
        model_center[:2] = 0
        self.model_global_info['model_center'] = model_center
        log.debug('mesh_paths=%r', mesh_paths)

        if isinstance(self.plane_info, ModelInfo):
            mesh_path = 'assets/models/default/plane.obj'
            mesh_info_path = 'assets/models/default/plane_objinfo.npy'
            mesh_info = np.load(mesh_info_path, allow_pickle=True).item()
            mesh_info['mesh_path'] = mesh_path
            mesh_name = util.to_camel_case(os.path.basename(mesh_path).split('.')[0])
            mesh_info['mesh_name'] = mesh_name
            self.plane_info.update_params(**mesh_info)

        model_name = "".join(mesh_names)
        return model_name
    
    def _sort_model_by_xy_len(self, model_dict):

        def update_position(info_dict, axis, val):
            if not info_dict['child_model']:
                info_dict[f'{axis}_pos'] += val
            else:
                update_position(info_dict['child_model'][0], axis, val)
                update_position(info_dict['child_model'][1], axis, val)

        n = len(model_dict)
        new_key = n
        while n > 1:
            swapped = False
            x_seq = list(range(n))
            y_seq = list(range(n))
            key = list(model_dict.keys())
            for i in range(2):
                for j in range(0, n-i-1):
                    if model_dict[key[x_seq[j]]]['x_len'] < model_dict[key[x_seq[j+1]]]['x_len']:
                        swapped = True
                        x_seq[j], x_seq[j+1] = x_seq[j+1], x_seq[j]
                if not swapped:
                    break
            for i in range(2):
                for j in range(0, n-i-1):
                    if model_dict[key[y_seq[j]]]['y_len'] < model_dict[key[y_seq[j+1]]]['y_len']:
                        swapped = True
                        y_seq[j], y_seq[j+1] = y_seq[j+1], y_seq[j]
                if not swapped:
                    break
            
            x_len_sum_min = model_dict[key[x_seq[-1]]]['x_len'] + model_dict[key[x_seq[-2]]]['x_len']
            y_len_sum_min = model_dict[key[y_seq[-1]]]['y_len'] + model_dict[key[y_seq[-2]]]['y_len']
            if x_len_sum_min < y_len_sum_min:
                key_1, key_2 = key[x_seq[-2]], key[x_seq[-1]]
                axis = 'x'
            else:
                key_1, key_2 = key[y_seq[-2]], key[y_seq[-1]]
                axis = 'y'

            update_position(model_dict[key_1], axis, -model_dict[key_2][f'{axis}_len']/2)
            update_position(model_dict[key_2], axis, model_dict[key_1][f'{axis}_len']/2)
    
            if axis == 'x':
                model_dict[new_key] = {
                    'x_len': model_dict[key_1][f'{axis}_len']+model_dict[key_2][f'{axis}_len'],
                    'y_len': max(model_dict[key_1][f'{axis}_len'], model_dict[key_2][f'{axis}_len']),
                    'x_pos': 0, 'y_pos': 0,
                    'model_index': new_key,
                    'child_model': [model_dict.pop(key_1), model_dict.pop(key_2)],
                }
            elif axis == 'y':
                model_dict[new_key] = {
                    'x_len': max(model_dict[key_1][f'{axis}_len'], model_dict[key_2][f'{axis}_len']),
                    'y_len': model_dict[key_1][f'{axis}_len']+model_dict[key_2][f'{axis}_len'],
                    'x_pos': 0, 'y_pos': 0,
                    'model_index': new_key,
                    'child_model': [model_dict.pop(key_1), model_dict.pop(key_2)],
                }
            else:
                raise ValueError()
            
            n = len(model_dict)
            new_key += 1

    def _sort_model_by_x_axis(self, model_dict):

        def update_position(info_dict, axis, val):
            if not info_dict['child_model']:
                info_dict[f'{axis}_pos'] += val
            else:
                update_position(info_dict['child_model'][0], axis, val)
                update_position(info_dict['child_model'][1], axis, val)

        n = len(model_dict)
        new_key = n
        while n > 1:
            swapped = False
            x_seq = list(range(n))
            key = list(model_dict.keys())
            for i in range(2):
                for j in range(0, n-i-1):
                    if model_dict[key[x_seq[j]]]['x_len'] < model_dict[key[x_seq[j+1]]]['x_len']:
                        swapped = True
                        x_seq[j], x_seq[j+1] = x_seq[j+1], x_seq[j]
                if not swapped:
                    break
            
            key_1, key_2 = key[x_seq[-2]], key[x_seq[-1]]
            axis = 'x'
            update_position(model_dict[key_1], axis, -model_dict[key_2][f'{axis}_len']/2)
            update_position(model_dict[key_2], axis, model_dict[key_1][f'{axis}_len']/2)
    
            model_dict[new_key] = {
                'x_len': model_dict[key_1][f'{axis}_len']+model_dict[key_2][f'{axis}_len'],
                'y_len': max(model_dict[key_1][f'{axis}_len'], model_dict[key_2][f'{axis}_len']),
                'x_pos': 0, 'y_pos': 0,
                'model_index': new_key,
                'child_model': [model_dict.pop(key_1), model_dict.pop(key_2)],
            }
            
            n = len(model_dict)
            new_key += 1

    def _iter_update_mesh_transform(self, model_dict, model_info):
        if not model_dict['child_model']:
            idx = model_dict['model_index']
            model_info[idx].transform[0, 3] = model_dict['x_pos']
            model_info[idx].transform[1, 3] = model_dict['y_pos']
            model_info[idx].transform = T(model_info[idx].transform)
        else:
            self._iter_update_mesh_transform(model_dict['child_model'][0], model_info)
            self._iter_update_mesh_transform(model_dict['child_model'][1], model_info)

    def _arrange_mesh_positions(self, is_superposition):
        objs = [futil.read_obj_mesh(model.mesh_path) for model in self.model_info]
        x_lens, y_lens = np.zeros((len(self.model_info))), np.zeros((len(self.model_info)))
        model_dict = {}
        for idx, obj in enumerate(objs):
            rot_ang_XY = np.random.uniform(0, 359)
            self.model_info[idx].transform = T.rotate([0,0,1], rot_ang_XY).matrix
            R = np.asarray(self.model_info[idx].transform)[:3, :3]
            obj_new = np.einsum('ij, ...j', R, obj)
            x_max, x_min = obj_new[:,0].max(), obj_new[:,0].min()
            y_max, y_min = obj_new[:,1].max(), obj_new[:,1].min()
            x_lens[idx], y_lens[idx] = x_max-x_min, y_max-y_min
            model_dict[idx] = {
                'x_len': obj_new[:,0].max() - obj_new[:,0].min(),
                'y_len': obj_new[:,1].max() - obj_new[:,1].min(),
                'x_pos': -(x_max+x_min)/2, 'y_pos': -(y_max+y_min)/2,
                'model_index': idx,
                'child_model': None,
            }
        if is_superposition:
            for i in model_dict:
                self._iter_update_mesh_transform(model_dict[i], self.model_info)
        else:
            if len(model_dict) > 1:
                self._sort_model_by_xy_len(model_dict)
            
            self._iter_update_mesh_transform(list(model_dict.values())[0], self.model_info)

    def _arrange_mesh_along_x(self):
        objs = [futil.read_obj_mesh(model.mesh_path) for model in self.model_info]
        x_lens, y_lens = np.zeros((len(self.model_info))), np.zeros((len(self.model_info)))
        model_dict = {}
        for idx, obj in enumerate(objs):
            rot_ang_XY = np.random.uniform(0, 359)
            self.model_info[idx].transform = T.rotate([0,0,1], rot_ang_XY).matrix
            R = np.asarray(self.model_info[idx].transform)[:3, :3]
            obj_new = np.einsum('ij, ...j', R, obj)
            x_max, x_min = obj_new[:,0].max(), obj_new[:,0].min()
            y_max, y_min = obj_new[:,1].max(), obj_new[:,1].min()
            x_lens[idx], y_lens[idx] = x_max-x_min, y_max-y_min
            model_dict[idx] = {
                'x_len': obj_new[:,0].max() - obj_new[:,0].min(),
                'y_len': obj_new[:,1].max() - obj_new[:,1].min(),
                'x_pos': -(x_max+x_min)/2, 'y_pos': -(y_max+y_min)/2,
                'model_index': idx,
                'child_model': None,
            }
        
        self._sort_model_by_x_axis(model_dict)
        
        self._iter_update_mesh_transform(list(model_dict.values())[0], self.model_info)


    def augment_mesh_info(self, is_mesh_superimpose=False):
        for _, model in enumerate(self.model_info):
            mesh_info = {}
            is_disp_allowed = not model.is_disp_prohibit
            mesh_info['is_disp'] = bool(is_disp_allowed*random.choice(self.mesh_disps))
            mesh_info['disp_strength'] = mesh_info['is_disp']*random.uniform(0.01, 0.2)
            mesh_info['smooth_level'] = random.choice(self.mesh_smooth_levels)
            mesh_info['smooth_level_scale'] = mesh_info['smooth_level']*random.uniform(0.65, 0.95)
            uv_transform = 1. / np.random.uniform(1.25, 20)
            uv_transform = (uv_transform, uv_transform)
            uv_transform_center = (np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9))
            mesh_info['uv_transform'] = uv_transform
            mesh_info['uv_transform_center'] = uv_transform_center
            model.update_params(**mesh_info)
        
        self._arrange_mesh_positions(is_mesh_superimpose)

        if isinstance(self.plane_info, ModelInfo):
            mesh_info = {}
            mesh_info['is_disp'] = random.choice([True, False])
            mesh_info['disp_strength'] = mesh_info['is_disp']*random.uniform(0.1, 0.3)
            mesh_info['smooth_level'] = 0
            mesh_info['smooth_level_scale'] = mesh_info['smooth_level']*random.uniform(0.65, 0.95)
            uv_transform = 1. / np.random.uniform(0.8, 2)
            uv_transform = (uv_transform, uv_transform)
            uv_transform_center = (np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9))
            mesh_info['uv_transform'] = uv_transform
            mesh_info['uv_transform_center'] = uv_transform_center
            scale_factor = np.random.uniform(1.0, 2.0)
            mesh_info['transform'] = T.scale([scale_factor, scale_factor, 1])
            self.plane_info.update_params(**mesh_info) 
            self.model_info.append(self.plane_info)       

    def augment_two_mesh_info(self):
        for _, model in enumerate(self.model_info):
            mesh_info = {}
            is_disp_allowed = not model.is_disp_prohibit
            mesh_info['is_disp'] = bool(is_disp_allowed*random.choice(self.mesh_disps))
            mesh_info['disp_strength'] = mesh_info['is_disp']*random.uniform(0.01, 0.2)
            mesh_info['smooth_level'] = random.choice(self.mesh_smooth_levels)
            mesh_info['smooth_level_scale'] = mesh_info['smooth_level']*random.uniform(0.65, 0.95)
            uv_transform = 1. / np.random.uniform(1.25, 20)
            uv_transform = (uv_transform, uv_transform)
            uv_transform_center = (np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9))
            mesh_info['uv_transform'] = uv_transform
            mesh_info['uv_transform_center'] = uv_transform_center
            model.update_params(**mesh_info)
        
        self._arrange_mesh_along_x()

        if isinstance(self.plane_info, ModelInfo):
            mesh_info = {}
            mesh_info['is_disp'] = random.choice([True, False])
            mesh_info['disp_strength'] = mesh_info['is_disp']*random.uniform(0.1, 0.3)
            mesh_info['smooth_level'] = 0
            mesh_info['smooth_level_scale'] = mesh_info['smooth_level']*random.uniform(0.65, 0.95)
            uv_transform = 1. / np.random.uniform(0.8, 2)
            uv_transform = (uv_transform, uv_transform)
            uv_transform_center = (np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9))
            mesh_info['uv_transform'] = uv_transform
            mesh_info['uv_transform_center'] = uv_transform_center
            scale_factor = np.random.uniform(1.0, 2.0)
            mesh_info['transform'] = T.scale([scale_factor, scale_factor, 1])
            self.plane_info.update_params(**mesh_info) 
            self.model_info.append(self.plane_info)       

    def _sample_material_dir(self, is_disp):
        attempt_counter = 0
        attempt_max = 200
        if self.material_paths:
            while(True):
                material_split_path = random.choice(self.material_paths)
                material_dirs = util.list_dir(material_split_path)
                if material_dirs:
                    material_dir = random.choice(material_dirs)
                    if is_disp:
                        m_name = os.path.split(material_dir)[-1]
                        disp_path = os.path.join(material_dir, f'{m_name}_4K_Displacement.jpg')
                        if os.path.isfile(disp_path):
                            break
                        else:
                            attempt_counter += 1
                            if attempt_counter >= attempt_max:
                                raise FileNotFoundError('Disp is required but not provided in material dir')
                            continue
                    else:
                        break
                else:
                    raise FileNotFoundError(f'Material directory not found in {material_split_path}')
        else:
            material_dir = ''
            print('Using the default material setting. '
                  'Disp is disabled')
        log.debug('material_dir=%r', material_dir)
        return material_dir

    def sample_material(self):
        assert self.model_info[0].mesh_path
        model_num = len(self.model_info)
        if model_num == 5:
            brdfs = random.sample(self.brdfs_ext, k=model_num)
        elif model_num <= 4:
            brdfs = random.sample(self.brdfs, k=model_num)
        else:
            raise ValueError(f'sample_material does not support model_num={model_num} (max 5)')
        log.debug('brdfs=%r', brdfs)
        for idx, model in enumerate(self.model_info):
            model.brdf = brdfs[idx]
            if 'pplastic' in model.brdf or 'dielectric' in model.brdf:
                model.ior = np.random.uniform(1.30, 1.70)
                model.conductor = None
            elif 'conductor' in model.brdf:
                model.ior = None
                model.conductor = random.choice(self.conductor_materials)
            else:
                raise ValueError(f'Unsupported BRDF: {model.brdf!r}')
            material_dir = self._sample_material_dir(model.is_disp)
            disp, albedo, rough = util.load_material(material_dir)
            if (not model.is_disp) or (not material_dir):
                disp = None
            if model.brdf in ['roughconductor', 'conductor', 'dielectric']:
                albedo = None
            if model.brdf in ['conductor', 'dielectric']:
                rough = None
            model.disp = disp
            model.albedo = albedo
            model.rough = rough

    def sample_2kinds_material(self):
        assert self.model_info[0].mesh_path
        model_num = len(self.model_info)
        if model_num == 2:
            brdfs = random.sample(('conductor', 'roughconductor'), k=1)+random.sample(('pplastic', 'roughpplastic'), k=1)
        else:
            raise ValueError(f'sample_2kinds_material requires exactly 2 models, got {model_num}')

        log.debug('brdfs=%r', brdfs)
        for idx, model in enumerate(self.model_info):
            model.brdf = brdfs[idx]
            if 'pplastic' in model.brdf or 'dielectric' in model.brdf:
                model.ior = np.random.uniform(1.30, 1.70)
                model.conductor = None
            elif 'conductor' in model.brdf:
                model.ior = None
                model.conductor = random.choice(self.conductor_materials)
            else:
                raise ValueError(f'Unsupported BRDF: {model.brdf!r}')
            material_dir = self._sample_material_dir(model.is_disp)
            disp, albedo, rough = util.load_material(material_dir)
            if (not model.is_disp) or (not material_dir):
                disp = None
            if model.brdf in ['roughconductor', 'conductor', 'dielectric']:
                albedo = None
            if model.brdf in ['conductor', 'dielectric']:
                rough = None
            model.disp = disp
            model.albedo = albedo
            model.rough = rough
    
    def augment_material(self):
        for model in self.model_info:
            rot_angle = np.random.randint(0, 360)
            rot_center = (np.random.uniform(0.2,0.8), np.random.uniform(0.2,0.8))
            print(f'{rot_angle=}', f'{rot_center=}')
            disp = rutil.rotate_image(model.disp, rot_angle, rot_center)
            albedo = rutil.rotate_image(model.albedo, rot_angle, rot_center)
            roughness = rutil.rotate_image(model.rough, rot_angle, rot_center)

            albedo, albedo_hsv_augment = rutil.color_augmentor(albedo)
            roughness, roughness_mean = rutil.roughness_augmentor(roughness, model.brdf)

            disp = rutil.to_sgl_chan_img(disp)
            roughness = rutil.to_sgl_chan_img(roughness)
            
            model.disp = disp
            model.albedo = albedo
            model.rough = roughness
            model.material_augment = {
                'rot_angle': rot_angle,
                'rot_center': rot_center,
                'albedo_hsv': albedo_hsv_augment,
                'roughness_mean':  roughness_mean,
            }
    
    def cache_material(self):
        for idx, model in enumerate(self.model_info):
            cache_dir = os.path.join(self.cache_dir, f"{idx}_{model.mesh_name}")
            os.makedirs(cache_dir, exist_ok=True)
            model.cache_dir = cache_dir
            model.clear_cache()
            img_names = ['disp', 'albedo', 'rough']
            for img_name in img_names:
                img = getattr(model, img_name)
                if img is not None:
                    save_path = os.path.join(cache_dir, f'{img_name}.png')
                    cv2.imwrite(save_path, img)
                    setattr(model, f'{img_name}_path', os.path.join(cache_dir, f'{img_name}.png'))

    def get_model_info(self, type='bl'):
        if type == 'bl':
            keys = [
                'mesh_name', 'mesh_path', 'transform', 'disp_strength', 'smooth_level', 'smooth_level_scale',
                'vertex_num', 'uv_transform', 'uv_transform_center', 'brdf',
                'disp_path', 'albedo_path', 'rough_path', 'cache_dir',
            ]
        elif type == 'mi':
            keys = ['brdf', 'albedo_path', 'rough_path', 'ior', 'conductor', 'cache_dir', ]
        else:
            raise ValueError(f"Unknown info type {type!r}; expected 'bl' or 'mi'")
        
        model_info = []
        to_ndarray_keys = ['transform']
        for m in self.model_info:
            info_dict = {}
            for key in keys:
                if key in to_ndarray_keys and type == 'bl':
                    val = np.array(getattr(m, key).matrix)
                else:
                    val = getattr(m, key)
                info_dict[key] = val
            model_info.append(info_dict)
        
        return model_info
    
    def get_material_model_info(self, material, type='bl'):
        if type == 'bl':
            keys = [
                'mesh_name', 'mesh_path', 'transform', 'disp_strength', 'smooth_level', 'smooth_level_scale',
                'vertex_num', 'uv_transform', 'uv_transform_center', 'brdf',
                'disp_path', 'albedo_path', 'rough_path', 'cache_dir',
            ]
        elif type == 'mi':
            keys = ['brdf', 'albedo_path', 'rough_path', 'ior', 'conductor', 'cache_dir', ]
        else:
            raise ValueError(f"Unknown info type {type!r}; expected 'bl' or 'mi'")
        
        model_info = []
        to_ndarray_keys = ['transform']
        for m in self.model_info:
            if material in getattr(m, 'brdf'):
                info_dict = {}
                for key in keys:
                    if key in to_ndarray_keys and type == 'bl':
                        val = np.array(getattr(m, key).matrix)
                    else:
                        val = getattr(m, key)
                    info_dict[key] = val
                
                model_info.append(info_dict)
            else:
                continue
        
        return model_info
    
    def summary_info(self):
        summary_dict = {}
        for idx, m in enumerate(self.model_info):
            m.albedo, m.disp, m.rough = None, None, None
            summary_dict[f'model_{idx}'] = vars(m)

        return summary_dict

    def clear(self):
        if os.path.isdir(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir)

class CameraInfo():
    def __init__(self):
        self.fov = 20
        self.resolution = (512, 512)
        self.transform = None
        self.sp_type = 'stratified'
        self.sp_num = 81
        self.rfilter = 'mitchell'
        self.view_dir = None
        self.up_axis = 'Z'

    def calc_transform(self, origin, target, up=[0,0,1]):
        self.transform = T.look_at(origin=origin, target=target, up=up)

    def update_view_dir(self, view_dir=None):
        if view_dir is None: 
            self.view_dir = np.array([0,0,-1]) # Mitsuba default camera view dir: [0,0,-1]
            if self.transform is not None:
                self.view_dir = np.einsum('ij,j', np.asarray(self.transform[:3,:3]), self.view_dir)
        else:
            self.view_dir = view_dir                

    def update_params(self, **kwargs):
        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, val)

class SceneSampler():
    """Sampler of rendering parameters of the scene: light condition and camera pose
    """
    def __init__(self, hdri_txt_path):
        self.hdri_paths = sutil.load_path_text(hdri_txt_path)
        self.lights = ('envmap', 'envmap_otherlights', 'otherlights')
        self.otherlight_types = ('point', 'spot', 'directional', 'projector')
        self.otherlight_nums = (1, 2, 3)
        
        self.cam_pose_sampling_methods = ('fibonacci_spiral', 'handcrafted')
        self.scene_dup_num = 1

        self.model_info = {}
        self.light_info = {}
        self.cam_info = []

    def load_model_info(self, model_center=[0,0,0], is_model_symmetric=False, is_model_flat=False):
        self.model_info['center'] = np.array(model_center)
        self.model_info['is_symmetric'] = is_model_symmetric
        self.model_info['is_flat'] = is_model_flat

    def sample_light_type(self):
        light_type = 'envmap'
        log.debug('light_type=%r', light_type)
        self.scene_dup_num = 1
        if light_type == 'envmap':
            if self.hdri_paths:
                hdri_path = random.choice(self.hdri_paths)
                log.debug('hdri_path=%r', hdri_path)
            else:
                print('Empty hdri path, using the constant env illumination!')
                hdri_path = ''
            env_int_scale = np.random.uniform(0.50,0.70)
            other_lt_type, other_lt_num = None, 0
        elif light_type == 'envmap_otherlights':
            if self.hdri_paths:
                hdri_path = random.choice(self.hdri_paths)
            else:
                print('Empty hdri path, using the constant env illumination!')
                hdri_path = ''
            env_int_scale = np.random.uniform(0.90,1.10)
            other_lt_type = random.choice(self.otherlight_types)
            other_lt_num = random.choice(self.otherlight_nums)
            log.debug('hdri_path=%r', hdri_path)
            log.debug('other_lt_type=%r', other_lt_type)
            log.debug('other_lt_num=%r', other_lt_num)
        elif light_type == 'otherlights':
            other_lt_type = random.choice(self.otherlight_types)
            other_lt_num = 1
            hdri_path, env_int_scale = None, None
            log.debug('other_lt_type=%r', other_lt_type)
            log.debug('other_lt_num=%r', other_lt_num)
            self.scene_dup_num = other_lt_num
        else:
            raise ValueError(f'Invalid light type "{light_type}"')
        
        self.light_info = {
            'light_type': light_type,
            'hdri_path': hdri_path,
            'hdri_scale': env_int_scale,
            'otherlight_type': other_lt_type,
            'otherlight_num': other_lt_num,
        }

    def sample_other_light(self):
        assert self.model_info
        assert self.light_info
        assert self.cam_info
        light_type = self.light_info['light_type']
        if 'otherlights' in light_type:
            other_lt_num = self.light_info['otherlight_num']
            other_lt_type = self.light_info['otherlight_type']
            view_num = len(self.cam_info['fovs'])
            model_center = self.model_info['center']
            if light_type == 'envmap_otherlights':
                scene_other_lt_dirs = [rutil.sample_envmap_otherlight_dirs(other_lt_num)]
            elif light_type == 'otherlights':
                cam_view_dirs_dup = rutil.duplicate_view_dirs(self.cam_other_info['view_dirs'], other_lt_num)
                scene_other_lt_dirs = []
                for i in range(view_num):
                    rutil.sample_otherlight_dirs(scene_other_lt_dirs, other_lt_num, cam_view_dirs_dup[i], True)
            else:
                raise NotImplementedError()
            
            if other_lt_type == 'point':
                other_lt_infos = rutil.get_point_light_info(scene_other_lt_dirs, model_center, light_type)
            elif other_lt_type == 'spot':
                other_lt_infos = rutil.get_spot_light_info(scene_other_lt_dirs, model_center, light_type)
            elif other_lt_type == 'projector':
                other_lt_infos = rutil.get_projector_light_info(scene_other_lt_dirs, model_center, light_type)
            elif other_lt_type == 'directional':
                other_lt_infos = rutil.get_directional_light_info(scene_other_lt_dirs, light_type)
            else:
                raise TypeError(f'invalid type of other lights: "{other_lt_type}"')
            
            self.light_info['otherlights'] = other_lt_infos
        else:
            pass

    def sample_cam_pose(self, pose_num, view_distance=15, fov=20, resolution=(512,512), sp_type='stratified', sp_num=81, rfilter='mitchell'):
        assert self.model_info
        assert self.light_info

        if self.model_info['is_symmetric']:
            sample_num = max(min(pose_num//4, 4), 1)
            if self.light_info['light_type'] == 'otherlights':
                sample_num = 1
        else:
            sample_num = pose_num
        
        self.cam_info = [CameraInfo() for _ in range(sample_num)]

        sample_method = random.choice(self.cam_pose_sampling_methods)
        if sample_num == 1:
            sample_method = 'handcrafted'
        if self.model_info['is_flat']:
            cam_view_dirs = rutil.flat_obj_sampling(sample_num)
        elif sample_method == 'fibonacci_spiral':
            cam_view_dirs = rutil.hemisphere_spiral_sampling(sample_num)
        elif sample_method == 'handcrafted':
            cam_view_dirs = rutil.handcrafted_sampling(sample_num)
        else:
            raise ValueError(f'Unsupported cam pose sampling method: {sample_method!r}')
        model_center = self.model_info['center']
        cam_origins = model_center[None] + view_distance * cam_view_dirs
        
        for i in range(sample_num):
            self.cam_info[i].calc_transform(cam_origins[i], model_center)
            self.cam_info[i].update_view_dir(cam_view_dirs[i])
            cam_params = {
                'fov':fov, 'resolution':resolution, 
                'sp_type':sp_type, 'sample_num':sp_num, 'rfilter':rfilter,
            }
            self.cam_info[i].update_params(**cam_params)

    def sample_cam_pose_along_y(self, pose_num, view_distance=15, fov=20, resolution=(512,512), sp_type='stratified', sp_num=81, rfilter='mitchell'):
        assert self.model_info
        assert self.light_info

        if self.model_info['is_symmetric']:
            sample_num = max(min(pose_num//4, 4), 1)
            if self.light_info['light_type'] == 'otherlights':
                sample_num = 1
        else:
            sample_num = pose_num
        
        self.cam_info = [CameraInfo() for _ in range(sample_num)]

        cam_view_dirs = rutil.alone_y_sampling(sample_num)
        model_center = self.model_info['center']
        cam_origins = model_center[None] + view_distance * cam_view_dirs
        
        for i in range(sample_num):
            self.cam_info[i].calc_transform(cam_origins[i], model_center)
            self.cam_info[i].update_view_dir(cam_view_dirs[i])
            cam_params = {
                'fov':fov, 'resolution':resolution, 
                'sp_type':sp_type, 'sample_num':sp_num, 'rfilter':rfilter,
            }
            self.cam_info[i].update_params(**cam_params)

    def get_light_info(self):
        light_type = self.light_info['light_type']
        light_info = {
            'light_type': light_type,
            'envmap': None,
            'otherlights': [None]*self.get_scene_num(),
        }
        if 'envmap' in light_type:
            light_info['envmap'] = {
                'hdri_path': self.light_info['hdri_path'],
                'hdri_scale': self.light_info['hdri_scale'],
            }
        if 'otherlights' in light_type:
            light_info['otherlights'] = self.light_info['otherlights']

        return light_info

    def get_cam_info(self, type='bl'):
        if type == 'bl':
            keys = ['fov', 'resolution', 'transform', 'up_axis']
        elif type == 'mi':
            keys = ['transform', 'fov', 'resolution', 'sp_type', 'sp_num', 'rfilter']
        else:
            raise ValueError(f"Unknown info type {type!r}; expected 'bl' or 'mi'")
        
        cam_info = []
        to_ndarray_keys = ['transform']
        for cam in self.cam_info:
            info_dict = {}
            for key in keys:
                if key in to_ndarray_keys and type == 'bl':
                    val = np.array(getattr(cam, key).matrix)
                else:
                    val = getattr(cam, key)
                info_dict[key] = val
            cam_info.append(info_dict)
        
        return cam_info
    
    def get_scene_dup_num(self):
        return self.scene_dup_num
    
    def get_scene_num(self):
        if self.light_info['light_type'] == 'otherlights':
            return len(self.cam_info) * self.light_info['otherlight_num']
        else:
            return 1
        
    def summary_info(self):
        summary_dict = {}
        summary_dict['global_model'] = self.model_info
        summary_dict['light'] = self.light_info
        for idx, caminfo in enumerate(self.cam_info):
            summary_dict[f'camera_{idx}'] = vars(caminfo)
            
        return summary_dict