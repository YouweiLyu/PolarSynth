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
        self.model_smooth_levels = (0, 0, 0, 0)
        
        self.brdfs = ('conductor', 'roughconductor', 'pplastic', 'roughpplastic')
        self.material_paths = sutil.load_path_text(material_txt_path)
        self.hdri_paths = sutil.load_path_text(hdri_txt_path)
        
        self.conductor_materials = (
            'a-C', 'Ag', 'Al', 'AlAs', 'Au', 'Be', 'Cr', 'Cu', 'Cu2O', 
            'Hg', 'HgTe', 'Ir', 'K', 'Li', 'MgO', 'Mo', 'Na_palik', 'Nb', 'Ni_palik', 'Rh', 
            'Se', 'SiC', 'Ta', 'Te', 'TiC', 'TiN', 'TiO2', 'VC', 'V_palik', 'VN', 'W',
        )
        self.lights = ['envmap']
        self.smooth_mtl_lights = ['envmap']
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
        
    def sample_model_augment_info(self):
        self.model_info['is_disp'] = False
        self.model_info['disp_strength'] = self.model_info['is_disp']*random.uniform(0.01, 0.2)
        self.model_info['smooth_level'] = random.choice(self.model_smooth_levels)
        self.model_info['smooth_level_scale'] = self.model_info['smooth_level']*random.uniform(0.65, 0.95)
        self.model_info['transform'] = T.translate([0, 0, 0])
        uv_transform = 1. / np.random.uniform(1.25, 20)
        uv_transform = (uv_transform, uv_transform)
        uv_transform_center = (np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9))
        uv_transform, uv_transform_center = (1,1), (0.5,0.5)
        print(f'{uv_transform=}\t{uv_transform_center=}')
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

    def assign_material(self, material, albedo_path, rough_value):
        assert self.model_info
        self.material_info.clear()
        self.brdf = material.split(':')[0]
        log.debug('self.brdf=%r', self.brdf)
        if 'pplastic' in self.brdf or 'dielectric' in self.brdf:
            model_ior = 1.50
            conductor_material = None
        elif 'conductor' in self.brdf:
            model_ior = None
            conductor_material = material.split(':')[-1]
        else:
            raise ValueError(f'Unsupported BRDF: {self.brdf!r}')
        self.material_info['ior'] = model_ior
        self.material_info['conductor'] = conductor_material
        disp, albedo, rough = None, None, None
        if albedo_path is not None: 
            albedo = cv2.imread(albedo_path, -1)
        if rough_value is not None:
            rough = np.ones((512, 512), dtype=np.uint8)*rough_value*255
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
        light_type = 'envmap'
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
    
    def assign_light_type(self, light_type, envmap_path):
        log.debug('light_type=%r', light_type)
        if light_type == 'envmap':
            if os.path.isfile(envmap_path):
                hdri_path = envmap_path
            else:
                print('Empty hdri path, using the constant env illumination!')
                hdri_path = ''
            env_int_scale = 0.6 # hardcoded
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

    def assign_cam_pose(self, pose_num, view_distance=15, fov=20, resolution=(512,512), sp_type='stratified', sp_num=81, rfilter='mitchell'):
        assert self.model_info
        assert self.light_info
        if self.model_info['is_symmetric']:
            sample_num = max(min(pose_num//4, 4), 1)
            if self.light_info['light_type'] == 'otherlights':
                sample_num = 1
        else:
            sample_num = pose_num
        sample_num = 1
        cam_pose_sampling = 'handcrafted'
        if self.model_info['is_flat']:
            cam_view_dirs = rutil.flat_obj_sampling(sample_num)
        elif cam_pose_sampling == 'fibonacci_spiral':
            cam_view_dirs = rutil.hemisphere_spiral_sampling(sample_num)
        elif cam_pose_sampling == 'handcrafted':
            theta = np.array([90])
            phi = np.deg2rad(0)
            x, y, z = np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)
            cam_view_dirs = np.stack([x, y, z], axis=-1)
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