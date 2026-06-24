import logging
import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import mitsuba as mi

os.environ['MI_DEFAULT_VARIANT'] = 'llvm_spectral_polarized'

from mitsuba import ScalarTransform4f as T

log = logging.getLogger(__name__)

def stks2polar(angle, stokes):
    """Convert stokes vector to polarization image given polarizer angle
    Args:
        angle: polarizer angle in degree
        stokes: Stokes vector [..., 3]
    """
    a = np.deg2rad(angle)
    p_vec = np.array([1, np.cos(2*a), np.sin(2*a)]) * 0.5
    polar = (p_vec * stokes).sum(axis=-1)
    return polar

def add_noise(array, noise_normal_std):
    noise = np.random.normal(0.0, noise_normal_std, size=array.shape)
    array = (array + noise).clip(0, 1)
    return array

def gamma_corr(img,):
    img = img**(1/2.2)
    return img

def sample_light_dir_cartes(theta_min=0., theta_max=180., phi_min=0., phi_max=360.):
    theta = np.deg2rad(np.random.uniform(theta_min, theta_max))
    phi = np.deg2rad(np.random.uniform(phi_min, phi_max))
    dir = np.array([
        np.sin(theta)*np.cos(phi),
        np.sin(theta)*np.sin(phi),
        np.cos(theta),
    ])
    return dir

def sample_envmap_otherlight_dirs(light_num):
    """sample light directions in the sphere 
    default setting all otherlights in one scene 
        Args: light_num
        Returns: lt_dirs (light_num, 3)
    """
    lt_dirs = []
    while(len(lt_dirs) < light_num):
        dir_sp = sample_light_dir_cartes(theta_min=1,theta_max=90)
        if lt_dirs:
            lt_dir_array = np.stack(lt_dirs)
            if (dir_sp[None] * lt_dir_array).sum(axis=-1).max() < 0.5:
                lt_dirs.append(dir_sp)
        else:
            lt_dirs.append(dir_sp)
    lt_dirs = np.stack(lt_dirs)
    return lt_dirs

def sample_otherlight_dirs(light_dirs, light_num, ref_z=None, in_z_hemisphere=False):
    """
        Sample light direction within the hemisphere of ref_z
        Also ensure the light direction within the hemisphere of z-axis
    """
    threshold = np.cos(np.deg2rad(40))
    lt_dirs_tmp = []
    sample_num_max = 1e4
    iter_num_max = 100
    iter_num = 0
    while not lt_dirs_tmp and iter_num < iter_num_max:
        iter_num += 1
        sample_num = 0
        while(len(lt_dirs_tmp) < light_num):
            sample_num += 1
            dir_local = sample_light_dir_cartes(theta_max=50)
            if ref_z is not None:
                ref_y = np.cross(ref_z, dir_local) 
                ref_x = np.cross(ref_y, ref_z)
                R = np.array([ref_x, ref_y, ref_z]).T
                dir_world = R @ dir_local
            else:
                dir_world = dir_local

            dir_world = dir_world / (np.sqrt((dir_world**2).sum(axis=-1))+1e-6)
            # check light direction within the hemisphere of z-axis
            if in_z_hemisphere and dir_world[-1] < 0:
                continue
            
            if lt_dirs_tmp:
                lt_dir_array = np.stack(lt_dirs_tmp)
                if (dir_world[None] * lt_dir_array).sum(axis=-1).max(axis=0) < threshold:
                    lt_dirs_tmp.append(dir_world)
                    light_dirs.append(dir_world[None])
            else:
                lt_dirs_tmp.append(dir_world)
                light_dirs.append(dir_world[None])
            
            if sample_num >= sample_num_max:
                break
        
    if iter_num >= iter_num_max:
        print(lt_dirs_tmp)
        print(light_dirs)
        print(len(lt_dirs_tmp), light_num)
        raise ValueError(f'too large light request ({light_num}) makes sample light directions failed!')

def duplicate_cam_positions(cam_positions, dup_num):
    dup_positions = []
    for i in range(cam_positions.shape[0]):
        dup_position = np.stack([cam_positions[i]]*dup_num)
        dup_positions.append(dup_position)
    dup_position_array = np.concatenate(dup_positions, axis=0)
    return dup_position_array

def duplicate_view_dirs(cam_view_dirs, dup_num):
    dup_view_dirs = []
    for i in range(cam_view_dirs.shape[0]):
        dup_view_dir = np.stack([cam_view_dirs[i]]*dup_num)
        dup_view_dirs.append(dup_view_dir)
    dup_position_array = np.concatenate(dup_view_dirs, axis=0)
    return dup_position_array

def get_point_light_info(scene_other_lt_dirs, obj_center, light_type):
    scene_other_lt_infos = []
    for light_dir_array in scene_other_lt_dirs:
        other_lt_infos = []
        light_num = light_dir_array.shape[0]
        distances = np.random.randint(30, 40, (light_num, 1))
        positions = distances*light_dir_array+obj_center[None]
        if light_type == 'envmap_otherlights':
            intensities = np.random.uniform(3, 4, (light_num, 1))
        else:
            intensities = np.random.uniform(10, 20, (light_num, 1))
        for i in range(light_num):
            other_lt_infos.append({
                'type': 'point',
                'position': positions[i].tolist(),
                'intensity': {
                    'type': 'rgb',
                    'value': float(intensities[i]),}
            })
        scene_other_lt_infos.append(other_lt_infos)
        # log.debug('intensities=%r', intensities)
    return scene_other_lt_infos

def get_spot_light_info(scene_other_lt_dirs, obj_center, light_type):
    scene_other_lt_infos = []
    for light_dir_array in scene_other_lt_dirs:
        other_lt_infos = []
        light_num = light_dir_array.shape[0]
        distances = np.random.randint(30, 40, (light_num, 1))
        if light_type == 'envmap_otherlights':
            intensities = np.random.uniform(3.0, 3.5, (light_num, 1))
        else:
            intensities = np.random.uniform(9.0, 11.0, (light_num, 1))
        origins = distances * light_dir_array + obj_center[None]
        for i in range(light_num):
            other_lt_infos.append({
                'type': 'spot',
                'to_world': mi.ScalarTransform4f.look_at(
                    origin=origins[i].tolist(),
                    target=obj_center.tolist(),
                    up=[0, 0, 1],
                ),
                'intensity': {
                    'type': 'spectrum',
                    'value': float(intensities[i]),}
            }) 
        scene_other_lt_infos.append(other_lt_infos)
        # log.debug('intensities=%r', intensities)
    return scene_other_lt_infos

def get_projector_light_info(scene_other_lt_dirs, obj_center, light_type):
    scene_other_lt_infos = []
    for light_dir_array in scene_other_lt_dirs:
        other_lt_infos = []
        light_num = light_dir_array.shape[0]
        distances = np.random.randint(30, 40, (light_num, 1))
        if light_type == 'envmap_otherlights':
            intensities = np.random.uniform(3.0, 3.5, (light_num, 1))
        else:
            intensities = np.random.uniform(9.0, 11.0, (light_num, 1))
        positions = distances*light_dir_array+obj_center[None]
        fovs = np.random.uniform(30, 40, (light_num))
        for i in range(light_num):
            other_lt_infos.append({
                'type': 'projector',
                'fov': float(fovs[i]),
                'to_world': mi.ScalarTransform4f.look_at(
                    origin=positions[i].tolist(),
                    target=obj_center.tolist(),
                    up=[0, 0, 1],
                ),
                'irradiance': {
                    'type': 'rgb',
                    'value': float(intensities[i])},
                'scale': 1.0,
            }) 
        scene_other_lt_infos.append(other_lt_infos)
    return scene_other_lt_infos

def get_directional_light_info(scene_other_lt_dirs, light_type):
    scene_other_lt_infos = []
    for light_dir_array in scene_other_lt_dirs:
        other_lt_infos = []
        light_num = light_dir_array.shape[0]
        if light_type == 'envmap_otherlights':
            intensities = np.random.uniform(0.006, 0.007, (light_num, 1))
        else:
            intensities = np.random.uniform(0.020, 0.025, (light_num, 1))
        for i in range(light_num):
            other_lt_infos.append({
                'type': 'directional',
                'direction': (-light_dir_array[i]).tolist(),
                'irradiance': {
                    'type': 'rgb',
                    'value': float(intensities[i])},
            }) 
        scene_other_lt_infos.append(other_lt_infos)
    return scene_other_lt_infos

def load_other_lights(other_lt_info):
    lt_type = other_lt_info['type']
    lt_num = other_lt_info['intensities'].shape[0]
    other_lt_dicts = []
    for idx in range(lt_num):
        if lt_type == 'point':
            lt_dict = {
                'type': 'point',
                'position': other_lt_info['positions'][idx].tolist(),
                'intensity': {
                    'type': 'rgb',
                    'value': float(other_lt_info['intensities'][idx]),},
            }
        elif lt_type == 'spot':
            lt_dict = {
                'type': 'spot',
                'to_world': mi.ScalarTransform4f.look_at(
                    origin=other_lt_info['positions'][idx].tolist(),
                    target=other_lt_info['targets'][0].tolist(),
                    up=[0, 0, 1],
                ),
                'intensity': {
                    'type': 'spectrum',
                    'value': float(other_lt_info['intensities'][idx]),}
            }
        elif lt_type == 'projector':
            lt_dict = {
                'type': 'projector',
                'fov': float(other_lt_info['fovs'][idx]),
                'to_world': mi.ScalarTransform4f.look_at(
                    origin=other_lt_info['positions'][idx],
                    target=other_lt_info['targets'][0],
                    up=[0, 0, 1],
                ),
                'irradiance': {
                    'type': 'rgb',
                    'value': float(other_lt_info['intensities'][idx]),
                },
                'scale': 1.0,
            }
        elif lt_type == 'directional':
            lt_dict = {
                'type': 'directional',
                'direction': other_lt_info['directions'][idx].tolist(),
                'irradiance': {
                    'type': 'rgb',
                    'value': float(other_lt_info['intensities'][idx]),
                },
            }
        else:
            raise NotImplementedError()
        other_lt_dicts.append(lt_dict)
    return other_lt_dicts


def load_sensor_(transform, fov=30, resolution=(512,512), sp_type='stratified', sp_num=81, rfilter='tent'):
    return {
        'type': 'perspective',
        'fov': fov,
        'to_world': transform,
        'sampler': {
            'type': sp_type,
            'sample_count': sp_num,
        },
        'film': {
            'type': 'hdrfilm',
            'width': resolution[1],
            'height': resolution[0],
            'rfilter': {
                'type': rfilter,
            },
        },
    }

def load_sensors(origins, targets, up=[0, 0, 1], fov=30, h=512, w=512, sp_type='stratified', sp_num=81, rfilter='tent'):
    sensors = []
    for idx in range(len(origins)):
        transform = mi.ScalarTransform4f.look_at(origin=origins[idx], target=targets[idx], up=up)
        sensors.append(load_sensor_(transform, fov, (h, w), sp_type, sp_num, rfilter))
    return sensors
        
def load_sensor(r, phi, theta, target=[0, 0, 0], up=[0, 0, 1], fov=30, h=512, w=512, sp_type='stratified', sp_num=81, rfilter='tent'):
    # Apply two rotations to convert from spherical coordinates to world 3D coordinates.
    origin = T.rotate([0, 0, 1], phi).rotate([0, 1, 0], theta) @ mi.ScalarPoint3f([0, 0, r])
    transform = mi.ScalarTransform4f.look_at(origin=origin, target=target, up=up)
    return load_sensor_(transform, fov, (h, w), sp_type, sp_num, rfilter)

def load_integrator(integrator_kwargs):
    return {
        'type': 'aov',
        # 'aovs': 'depth:depth,normal:sh_normal,albedo:albedo',
        'aovs': 'normal:sh_normal,albedo:albedo',
        'stokes': {
            'type': 'stokes',
            'integrator': integrator_kwargs
        }
    }

def load_pplastic_model(
        albedo, roughness, model_path=None, normal_variant=None, 
        ior=1.4, dist='beckmann', model_type='sphere', 
        uv_transform=T.scale([1,1,0]), transform=None):
    model_dict = {'type': model_type,}
    bsdf_dict = {
        'type': 'pplastic',
        'diffuse_reflectance':{
            'type': 'bitmap',
            'wrap_mode': 'repeat',
            'to_uv': uv_transform,
        },
        'int_ior': ior,
        'distribution': dist,
        'alpha': {
            'type': 'bitmap',
            'wrap_mode': 'repeat',
            'to_uv': uv_transform,
        },
    }
    if isinstance(albedo, str):
        bsdf_dict['diffuse_reflectance']['filename'] = albedo
    elif isinstance(albedo, mi.Bitmap):
        bsdf_dict['diffuse_reflectance']['bitmap'] = albedo
    elif isinstance(albedo, float):
        bsdf_dict['diffuse_reflectance'] = albedo
    else:
        raise TypeError(f'Invalid type {type(albedo)} for albedo')
    if isinstance(roughness, str):
        bsdf_dict['alpha']['filename'] = roughness
    elif isinstance(roughness, mi.Bitmap):
        bsdf_dict['alpha']['bitmap'] = roughness
    elif isinstance(roughness, float):
        bsdf_dict['alpha'] = roughness
    else:
        raise TypeError(f'Invalid type {type(roughness)} for roughness')
    
    if model_path is not None:
        model_dict['filename'] = model_path
    if normal_variant is not None:
        model_dict['normalbsdf'] = {
            'type': 'normalmap',
            'normalmap': {
                'type': 'bitmap',
                'raw': True,
                'bitmap': normal_variant,
                'to_uv': uv_transform,
            },
            'nested_bsdf': bsdf_dict,
        }
    else:
        model_dict['bsdf'] = bsdf_dict
    if transform is not None:
        model_dict['to_world'] = transform
    
    return model_dict

def load_pplastic_model_bump(
        albedo, roughness, model_path=None, displacement=None, 
        ior=1.4, dist='beckmann', model_type='sphere', 
        uv_transform=T.scale([1,1,0]), transform=None):
    bsdf_dict = {
        'type': 'pplastic',
        'diffuse_reflectance':{
            'type': 'bitmap',
            'bitmap': albedo,
            'wrap_mode': 'repeat',
            'to_uv': uv_transform,
        },
        'int_ior': ior,
        'distribution': dist,
        'alpha': roughness,
    }
    model_dict = {
        'type': model_type,
    }
    if model_path is not None:
        model_dict['filename'] = model_path
    if displacement is not None:
        model_dict['bumpbsdf'] = {
            'type': 'bumpmap',
            'bumpmap': {
                'type': 'bitmap',
                'raw': True,
                'bitmap': displacement,
                'to_uv': uv_transform,
            },
            'nested_bsdf': bsdf_dict,
        }
    else:
        model_dict['bsdf'] = bsdf_dict
    if transform is not None:
        model_dict['to_world'] = transform
    return model_dict

def load_conductor_model(material, model_path=None, model_type='sphere', transform=None):
    model_dict = {
        'type': model_type,
        'bsdf': {
            'type': 'conductor',
            'material': material,}
    }
    if model_path is not None:
        model_dict['filename'] = model_path
    if transform is not None:
        model_dict['to_world'] = transform
    return model_dict

def load_roughconductor_model(material, roughness, model_path=None,
        dist='beckmann', model_type='sphere', 
        uv_transform=T.scale([1,1,0]), transform=None):
    model_dict = {'type': model_type,}
    bsdf_dict = {
        'type': 'roughconductor',
        'material': material,
        'distribution': dist,
        'alpha': {
            'type': 'bitmap',
            'wrap_mode': 'repeat',
            'to_uv': uv_transform,
        },
    }
    if isinstance(roughness, str):
        bsdf_dict['alpha']['filename'] = roughness
    elif isinstance(roughness, mi.Bitmap):
        bsdf_dict['alpha']['bitmap'] = roughness
    elif isinstance(roughness, float):
        bsdf_dict['alpha'] = roughness
    else:
        raise TypeError('Invalid type for roughness')
    model_dict['bsdf'] = bsdf_dict
    if model_path is not None:
        model_dict['filename'] = model_path
    if transform is not None:
        model_dict['to_world'] = transform
    return model_dict

def load_dielectric_model(ior, model_path=None, model_type='sphere', transform=None):
    model_dict = {
        'type': model_type,
        'bsdf': {
            'type': 'dielectric',
            'int_ior': ior,}
    }
    if model_path is not None:
        model_dict['filename'] = model_path
    if transform is not None:
        model_dict['to_world'] = transform
    return model_dict

def load_plane_pplastic(
        normal_variant=None, model_type='sphere', 
        uv_transform=T.scale([1,1,0]), transform=None):
    bsdf_dict = {
        'type': 'pplastic',
        'diffuse_reflectance':{
            'type': 'rgb',
            'value': 0.5
        },
        'specular_reflectance':{
            'type': 'rgb',
            'value': 0
        },
        'alpha': 0.5
    }
    model_dict = {
        'type': model_type,
    }
    if normal_variant is not None:
        model_dict['normalbsdf'] = {
            'type': 'normalmap',
            'normalmap': {
                'type': 'bitmap',
                'raw': True,
                'bitmap': normal_variant,
                'to_uv': uv_transform,
            },
            'nested_bsdf': bsdf_dict,
        }
    else:
        model_dict['bsdf'] = bsdf_dict
    if transform is not None:
        model_dict['to_world'] = transform
    return model_dict


def load_envmap(envmap_path, scale, rotate=False):
    if 'polyhaven' in envmap_path:
        dataset_coeff = 1
    elif 'ambientcg' in envmap_path:
        dataset_coeff = 1
    else:
        dataset_coeff = 1
        print('Warning: unrecognized envmap path type!')
    scaled_intensity = scale * dataset_coeff
    envmap_dict = {
        'type': 'envmap',
        'filename': envmap_path,
        'scale' : scaled_intensity,
    }
    if rotate:
        envmap_dict['to_world'] = T.rotate([1, 0, 0], 90)
    return envmap_dict

def load_const_env(int_val):
    const_dict = {
        'type': 'constant',
        'radiance': {
            'type': 'rgb',
            'value': int_val,
        }
    }
    return const_dict

def load_distant_light(intensity, direction):
    dist_lt_dict = {
        'type': 'directional',
        'direction': direction,
        'irradiance': {
            'type': 'rgb',
            'value': intensity,
        }
    }
    return dist_lt_dict

def rotate_image(image, angle, center=None):
    if image is not None:
        image_xy = image.shape[1::-1]
        if center is None:
            center_xy = tuple(np.array(image.shape[1::-1]) / 2)
        else:
            center_xy = (image_xy[0]*center[0], image_xy[1]*center[1])
        rot_mat = cv2.getRotationMatrix2D(center_xy, angle, 1.0)
        image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], 
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return image

def color_augmentor(bgr):
    hsv_dict = {}
    if bgr is not None:
        bgr = bgr.astype(np.float32) / 255.
        a_hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hsv_dict['hue_angle'] = np.random.uniform(-180, 180)
        hsv_dict['saturation_shift'] = np.random.uniform(-0.2, 0.2)
        hsv_dict['value_scale'] = np.random.uniform(0.8, 1.3)
        a_hsv[...,0] += hsv_dict['hue_angle']
        a_hsv[...,1] += hsv_dict['saturation_shift']
        a_hsv[...,2] *= hsv_dict['value_scale']
        a_hsv[...,1:] = a_hsv[...,1:].clip(0, 1)
        bgr = cv2.cvtColor(a_hsv, cv2.COLOR_HSV2BGR)
        bgr = (bgr.clip(0.0196,1) * 255.).astype(np.uint8)
    return bgr, hsv_dict

def roughness_augmentor(sigma, brdf):
    r_exp_mean = None
    if sigma is not None:
        sigma = sigma.astype(np.float32) / 255.
        if brdf in ['roughconductor', 'roughpplastic']:
            r = random.choice(['regular', 'rough'])
        elif brdf in ['pplastic']:
            r = random.choice(['smooth', 'regular'])
        else:
            raise ValueError(f'Invalid brdf type: {brdf}')
        
        if r == 'smooth':
            r_exp_mean = np.random.uniform(0.005, 0.05)
        elif r == 'regular':
            r_exp_mean = np.random.uniform(0.1, 0.3)
        else:
            r_exp_mean = np.random.uniform(0.3, 0.7)

        r_ratio = r_exp_mean / np.mean(sigma)
        sigma = ((sigma * r_ratio).clip(0, 1) * 255).astype(np.uint8)
    return sigma, r_exp_mean

def to_sgl_chan_img(img):
    if img is not None:
        if len(img.shape) == 3 and img.shape[-1] == 3:
            img = img[...,0].copy()
    return img

def augment_material(material_dict, brdf):
    disp = material_dict['displacement']
    albedo = material_dict['albedo_diff']
    roughness = material_dict['roughness']

    rot_angle = np.random.randint(0, 360)
    rot_center = (np.random.uniform(0.2,0.8), np.random.uniform(0.2,0.8))

    if brdf in ['roughconductor', 'conductor', 'dielectric']:
        albedo = None
    if brdf in ['conductor', 'dielectric']:
        roughness = None
    
    disp = rotate_image(disp, rot_angle, rot_center)
    albedo = rotate_image(albedo, rot_angle, rot_center)
    roughness = rotate_image(roughness, rot_angle, rot_center)

    albedo = color_augmentor(albedo)
    roughness = roughness_augmentor(roughness, brdf)

    uv_transform = 1. / np.random.uniform(1.25, 20)
    uv_transform = (uv_transform, uv_transform)
    uv_transform_center = (np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9))
    log.debug('uv_transform_center=%r', uv_transform_center)
    log.debug('uv_transform=%r', uv_transform)
    
    if disp is not None and len(disp.shape) == 3 and disp.shape[-1] == 3:
        disp = disp.mean(axis=-1)
    if roughness is not None and len(roughness.shape) == 3 and roughness.shape[-1] == 3:
        roughness = roughness.mean(axis=-1)

    return {
        'displacement': disp,
        # 'normal': normal,
        'albedo_diff': albedo,
        'roughness': roughness,
        'uv_transform': uv_transform,
        'uv_transform_center': uv_transform_center,
    }

def overlook_sampling(num_pts):
    theta = np.deg2rad(np.random.uniform(1, 10, (num_pts)))
    phi = np.deg2rad(np.linspace(0, 360, num_pts+1)[:num_pts])
    x, y, z = np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)
    position_array = np.stack([x, y, z], axis=-1)
    return position_array

def hemisphere_spiral_sampling(num_pts):
    indices = np.arange(0, num_pts, dtype=float) + 0.5
    sampling_pts = np.random.normal(0, 0.05, num_pts)
    indices = (indices + sampling_pts).clip(0, num_pts)
    theta = np.arccos(1 - indices/num_pts)
    phi = np.pi * (1 + 5**0.5) * indices
    x, y, z = np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)
    position_array = np.stack([x, y, z], axis=-1)
    return position_array

def handcrafted_sampling(num_pts):
    if num_pts == 8:
        theta = np.array([50, 30, 50, 30, 65, 90, 65, 90])
        theta_perturb = np.random.normal(0, 2, num_pts)
        theta = np.deg2rad((theta + theta_perturb).clip(0, 90))
        phi = np.array([0, 90, 180, 270, 45, 135, 225, 315])
        phi_perturb = np.random.normal(0, 5, num_pts)
        phi = np.deg2rad(phi + phi_perturb)
    elif num_pts == 4:
        theta = np.array([50, 90, 30, 70])
        theta_perturb = np.random.normal(0, 4, num_pts)
        theta = np.deg2rad((theta + theta_perturb).clip(0, 90))
        phi = np.array([0, 90, 180, 270])
        phi_perturb = np.random.normal(0, 5, num_pts)
        phi = np.deg2rad(phi + phi_perturb)
    elif num_pts == 2:
        theta = np.array([80, 60])
        theta_perturb = np.random.normal(0, 5, num_pts)
        theta = np.deg2rad((theta + theta_perturb).clip(0, 90))
        phi = np.array([0, 180])
        phi_perturb = np.random.normal(0, 10, num_pts)
        phi = np.deg2rad(phi + phi_perturb)
    elif num_pts == 1:
        # theta = np.array([90])
        # theta_perturb = np.random.normal(0, 10, num_pts)
        # theta = np.deg2rad((theta + theta_perturb).clip(0, 90))
        theta = np.deg2rad(np.random.uniform(30, 80, (num_pts)))
        phi = np.deg2rad(np.random.uniform(-180, 180, (num_pts)))
    else:
        raise Exception(f'Invalid sampling num: {num_pts}')
    
    x, y, z = np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)
    position_array = np.stack([x, y, z], axis=-1)
    return position_array

def alone_y_sampling(num_pts):
    if num_pts == 1:
        theta = np.deg2rad(np.random.uniform(40, 70, (num_pts)))
        phi = np.deg2rad(np.random.uniform(89, 91, (num_pts)))
    else:
        raise Exception(f'Invalid sampling num: {num_pts}')
    
    x, y, z = np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)
    position_array = np.stack([x, y, z], axis=-1)
    return position_array

def flat_obj_sampling(num_pts):
    theta = np.deg2rad(np.random.uniform(30, 55, (num_pts)))
    phi = np.linspace(0, 360, num_pts+1)[:num_pts]
    phi_perturb = np.random.normal(0, 5, num_pts)
    phi = np.deg2rad(phi + phi_perturb)
    x, y, z = np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)
    position_array = np.stack([x, y, z], axis=-1)
    return position_array

def cube_sampling():
    theta = np.deg2rad(np.array([1, 90, 90, 90, 90, 179]))
    phi = np.deg2rad(np.array([0, 0, 90, 180, 270, 180]))
    x, y, z = np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)
    position_array = np.stack([x, y, z], axis=-1)
    return position_array



