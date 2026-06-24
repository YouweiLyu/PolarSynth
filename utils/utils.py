import os
import cv2
import numpy as np
import mitsuba as mi
from mitsuba import ScalarTransform4f as T

def cycle(a_list):
    while True:
        for item in a_list:
            yield item

def vprint(v):
    print(f'{v=}')

def cache_cam_info(cam_positions, model_center, fov, res_h, res_w, info_dict):
    cam_world_mats = []
    cam_up_axes = []
    cam_resolutions = []
    fovs = []
    for i in range(cam_positions.shape[0]):
        cam_position = cam_positions[i]
        cam_world_mats.append(np.asarray(T.look_at(
            origin=cam_position,
            target=model_center,
            up=[0, 0, 1]
        ).matrix))
        cam_up_axes.append('Z')
        fovs.append(fov)
        cam_resolutions.append((res_h,res_w))
    
    info_dict['camera_fovs'] = fovs
    info_dict['camera_resolutions'] = cam_resolutions
    info_dict['camera_transforms'] = cam_world_mats
    info_dict['camera_up_axes'] = cam_up_axes

def cache_material(material_dict, info_dict, cache_dir='./tmp'):
    os.makedirs(cache_dir, exist_ok=True)
    img_names = ['displacement', 'albedo_diff', 'roughness']
    other_names = ['uv_transform', 'uv_transform_center']
    for img_name in img_names:
        if material_dict[img_name] is not None:
            img = material_dict[img_name]
            save_path = os.path.join(cache_dir, f'{img_name}.png')
            cv2.imwrite(save_path, img)
            info_dict[f'{img_name}_paths'] = [os.path.join(cache_dir, f'{img_name}.png')]
        else:
            info_dict[f'{img_name}_paths'] = [None]
    for other_name in other_names:
        info_dict[f'{other_name}s'] = [material_dict[other_name]]

def load_material(material_dir, res='4K', format='jpg'):
    if os.path.isdir(material_dir):
        material_name = os.path.split(material_dir)[-1]
        displace_path = os.path.join(material_dir, f'{material_name}_{res.upper()}_Displacement.{format}')
        color_path = os.path.join(material_dir, f'{material_name}_{res.upper()}_Color.{format}')
        rough_path = os.path.join(material_dir, f'{material_name}_{res.upper()}_Roughness.{format}')

        displace = cv2.imread(displace_path) if os.path.isfile(displace_path) else None
        color = cv2.imread(color_path) if os.path.isfile(color_path) else None
        rough = cv2.imread(rough_path, -1) if os.path.isfile(rough_path) else None
        if color is None:
            color = (np.stack([
                np.ones((512, 512))*np.random.uniform(0.1, 0.9),
                np.ones((512, 512))*np.random.uniform(0.1, 0.9),
                np.ones((512, 512))*np.random.uniform(0.1, 0.9),
            ], axis=-1) * 255).astype(np.uint8)
        if rough is None:
            rough = np.ones((512, 512), dtype=np.uint8) * 255
        
    else:
        displace = None
        color = np.ones((512, 512, 3)) * 255
        rough = np.ones((512, 512), dtype=np.uint8) * 255

    return displace, color, rough

def list_dir(dir_):
    dirs = []
    for item in os.listdir(dir_):
        if os.path.isdir(os.path.join(dir_, item)):
            dirs.append(os.path.join(dir_, item))
    return dirs

def to_camel_case(snake_str):
    return "".join(x.capitalize() for x in snake_str.lower().split("_"))

def to_lower_camel_case(snake_str):
    # We capitalize the first letter of each component except the first one
    # with the 'capitalize' method and join them together.
    camel_string = to_camel_case(snake_str)
    return snake_str[0].lower() + camel_string[1:]

def get_file_paths(file_txt_path, file_dir, suffix=''):
    with open(file_txt_path) as f:
        paths = f.read().splitlines()
    file_paths = [os.path.join(file_dir, file+suffix) for file in paths]
    return file_paths

def load_sensor(r, phi, theta):
    # Apply two rotations to convert from spherical coordinates to world 3D coordinates.
    origin = T.rotate([0, 0, 1], phi).rotate([0, 1, 0], theta) @ mi.ScalarPoint3f([0, 0, r])

    return mi.load_dict({
        'type': 'perspective',
        'fov': 39.3077,
        'to_world': T.look_at(
            origin=origin,
            target=[0, 0, 0],
            up=[0, 0, 1]
        ),
        'sampler': {
            'type': 'independent',
            'sample_count': 16
        },
        'film': {
            'type': 'hdrfilm',
            'width': 256,
            'height': 256,
            'rfilter': {
                'type': 'tent',
            },
            'pixel_format': 'rgb',
        },
    })