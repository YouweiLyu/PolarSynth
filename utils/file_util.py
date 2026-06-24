import os
import gc
import cv2
import json
import Imath
import shutil
import mitsuba as mi
import OpenEXR as openexr
import numpy as np
from six.moves import cPickle as pkl
import matplotlib
import matplotlib.pyplot as plt

from . import render_util as rutil

matplotlib.use('agg')
pt = Imath.PixelType(Imath.PixelType.FLOAT)

class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, mi.ScalarTransform4f):
            return np.asarray(obj.matrix).tolist()
        return json.JSONEncoder.default(self, obj)

def img_write(imgpath, array, **kwargs):
    if not cv2.imwrite(imgpath, array, **kwargs):
        raise Exception(f'Fail to save {imgpath}')

def img_read(path):
    img = cv2.imread(path, -1)
    depth = 65535. if img.dtype == np.uint16 else 255.
    img_norm = img.astype(np.float32) / depth
    return img_norm

def read_obj_mesh(path):
    if path.split('.')[-1].lower() == 'obj':
        x,y,z = [],[],[]
        with open(path, 'r') as f:
            obj_lines = f.readlines()
        for line in obj_lines:
            if line.startswith('v '):
                vertices = line.split(' ')[1:]
                x.append(float(vertices[0]))
                y.append(float(vertices[1]))
                z.append(float(vertices[2]))
        point_cloud = np.stack([np.asarray(x), np.asarray(y), np.asarray(z)],-1)
        return point_cloud
    else:
        raise ValueError(f'Invalid file type: {path.split(".")[-1].lower()}')

def make_dirs(dst_dir, subfolders):
    for folder in subfolders:
        os.makedirs(os.path.join(dst_dir, folder), exist_ok=True)

def make_save_dirs(dst_dir, save_items):
    subfolders = []
    item2folders = {
        'polar': ['pol000', 'pol045', 'pol090', 'pol135'],
        'material': ['material_tag'],
    }
    for item in save_items:
        if item in item2folders:
            subfolders.extend(item2folders[item])
        else:
            subfolders.append(item)
    subfolders = list(set(subfolders))
    make_dirs(dst_dir, subfolders)


def clear_files(dir_):
    """Clear all files in the directory
    """
    for file in os.listdir(dir_):
        if os.path.isfile(os.path.join(dir_, file)):
            os.remove(os.path.join(dir_, file))

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pkl.dump(di_, f, protocol=pkl.HIGHEST_PROTOCOL)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pkl.load(f)
    return ret_di

def ddepth2dtype(ddepth: int):
    if ddepth == 255:
        save_type = np.uint8
    elif ddepth == 65535:
        save_type = np.uint16
    else:
        raise ValueError(f'Invalid data depth "{ddepth}"')
    return save_type

def get_exr_cmpnt(exr_file, key, size_hw):
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    return np.fromstring(exr_file.channel(key, pt), dtype=np.float32).reshape(size_hw)


def exr_size_hw(exr_file):
    """Return ``(height, width)`` of an OpenEXR file, no re-decoded once cached."""
    dw = exr_file.header()['dataWindow']
    return (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)


def read_channels(exr_file, channels, size_hw=None):
    """Batch-decode several channels at once.

    Faster than calling :func:`get_exr_cmpnt` N times because the EXR header
    and PixelType objects are reused. Returns a list of arrays in the order
    requested, each shaped ``size_hw``.
    """
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    if size_hw is None:
        size_hw = exr_size_hw(exr_file)
    return [
        np.fromstring(exr_file.channel(c, pt), dtype=np.float32).reshape(size_hw)
        for c in channels
    ]

def save_img_mitsuba_exrpath(exr_path, save_path, ddepth=255):
    exr_file = openexr.InputFile(exr_path)
    save_img_mitsuba_exr(exr_file, save_path, ddepth)

def save_img_mitsuba_exr(exr_file, save_path, ddepth=255):
    save_type = ddepth2dtype(ddepth)
    r, g, b = read_channels(exr_file, ('R', 'G', 'B'))
    img = np.stack([b, g, r], -1)
    img = (img.clip(0, 1)**(1.0/2.2) * ddepth).astype(save_type)
    cv2.imwrite(save_path, img)

def save_albedo_mitsuba_exr(exr_file, save_path, ddepth=255):
    save_type = ddepth2dtype(ddepth)
    r, g, b = read_channels(exr_file, ('albedo.R', 'albedo.G', 'albedo.B'))
    img = np.stack([b, g, r], -1)
    img = (img.clip(0, 1) * ddepth).astype(save_type)
    cv2.imwrite(save_path, img)
    return img

def save_depth_mitsuba_exr(exr_file, save_path):
    (depth,) = read_channels(exr_file, ('depth',))
    d_max, d_min = depth[depth > 0].max(), depth[depth > 1e-6].min()
    img = (depth - d_min) / (d_max - d_min)
    img = (img * 255).clip(0, 255)
    cv2.imwrite(save_path, img)

def save_normal_mitsuba_exr(exr_file, save_path, ddepth=65535, world_mat=None, save_suffix='_normal_mi.png'):
    save_type = ddepth2dtype(ddepth)
    nx, ny, nz, depth = read_channels(
        exr_file, ('normal.X', 'normal.Y', 'normal.Z', 'depth.T')
    )
    normals = np.stack([nz, ny, nx], -1)
    normal_mod = (normals ** 2).sum(axis=-1, keepdims=True)
    # mask = (depth[..., None] > 0) * (normal_mod > 0.1)
    mask = depth[..., None] > 0
    normals = normals / (normal_mod+1e-6) * mask

    normal_img = ((normals + 1) / 2).clip(0, 1) * ddepth
    cv2.imwrite(save_path.replace(save_suffix, '_normal_G_mi.png'), normal_img.astype(save_type))
    cv2.imwrite(save_path.replace(save_suffix, '_mask_mi.png'), mask[...,0]*255)
    if world_mat is not None:
        _, R, _, _, _, _, _ = cv2.decomposeProjectionMatrix(world_mat[:3])
        
        normals_rot = np.einsum('ij,...j', np.linalg.inv(R), normals[...,::-1])
        normals_rot = normals_rot / ((normals_rot**2).sum(axis=-1, keepdims=True)+1e-6) * mask
        normals_rot = np.stack([-normals_rot[...,0], normals_rot[...,1], -normals_rot[...,2],], -1)
        normal_rot_img = ((normals_rot + 1) / 2).clip(0, 1) * ddepth
        normal_rot_img = normal_rot_img[...,::-1]
        cv2.imwrite(save_path.replace(save_suffix, '_normal_L_mi.png'), normal_rot_img.astype(save_type))
    # save_path = save_path.replace('_mask.png', '_norm.png')
    # cv2.imwrite(save_path, normal_mod*255)

def save_polarprop_exr(exr_file, save_path, eps=1e-6):
    chans = read_channels(exr_file, (
        'stokes.S0.R', 'stokes.S0.G', 'stokes.S0.B',
        'stokes.S1.R', 'stokes.S1.G', 'stokes.S1.B',
        'stokes.S2.R', 'stokes.S2.G', 'stokes.S2.B',
    ))
    s0 = np.stack(chans[0:3], axis=-1).mean(axis=-1)
    s1 = np.stack(chans[3:6], axis=-1).mean(axis=-1)
    s2 = np.stack(chans[6:9], axis=-1).mean(axis=-1)
    aolp, dolp = 0.5*np.arctan2(s2, s1+eps), np.sqrt(s1**2+s2**2)/(s0+eps)
    fig, ax = plt.subplots(1,2, figsize=(14,6), dpi=200)
    ax[0].axis('off')
    ax[0].set_title(f'DoLP')
    ax_0 = ax[0].imshow(dolp, cmap='GnBu', vmax=1, vmin=0)
    bar_0 = fig.colorbar(ax_0, ax=ax[0])
    ax[1].axis('off')
    ax[1].set_title(f'AoLP')
    ax_1 = ax[1].imshow(np.rad2deg(aolp), cmap='hsv', vmax=90, vmin=-90)
    bar_1 = fig.colorbar(ax_1, ax=ax[1])
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close('all')
    gc.collect()
    

def save_normal_blender_exr(exr_file, save_path, ddepth=65535, world_mat=None, save_suffix='_normal_bl.png'):
    save_type = ddepth2dtype(ddepth)
    nz, ny, nx, depth = read_channels(exr_file, (
        'ViewLayer.Normal.Z', 'ViewLayer.Normal.Y',
        'ViewLayer.Normal.X', 'ViewLayer.Depth.Z',
    ))
    normals = np.stack([nz, ny, nx], -1)
    normal_mod = (normals ** 2).sum(axis=-1, keepdims=True)
    # mask = (depth[..., None] > 0) * (normal_mod > 0.1)
    mask = depth[..., None] < 1e10
    normals = normals / (normal_mod+1e-6) * mask

    normal_img = ((normals + 1) / 2).clip(0, 1) * ddepth
    cv2.imwrite(save_path.replace(save_suffix, '_normal_G_bl.png'), normal_img.astype(save_type))
    cv2.imwrite(save_path.replace(save_suffix, '_mask_bl.png'), mask[...,0]*255)
    if world_mat is not None:
        _, R, _, _, _, _, _ = cv2.decomposeProjectionMatrix(world_mat[:3])
        normals_rot = np.einsum('ij,...j', np.linalg.inv(R), normals[...,::-1])
        normals_rot = normals_rot / ((normals_rot**2).sum(axis=-1, keepdims=True)+1e-6) * mask
        normals_rot = np.stack([-normals_rot[...,0], normals_rot[...,1], -normals_rot[...,2],], -1)
        normal_rot_img = ((normals_rot + 1) / 2).clip(0, 1) * ddepth
        normal_rot_img = normal_rot_img[...,::-1]
        cv2.imwrite(save_path.replace(save_suffix, '_normal_L_bl.png'), normal_rot_img.astype(save_type))

def save_albedo_blender_exr(exr_file, save_path, ddepth=255):
    save_type = ddepth2dtype(ddepth)
    b, g, r = read_channels(exr_file, (
        'ViewLayer.DiffCol.B', 'ViewLayer.DiffCol.G', 'ViewLayer.DiffCol.R',
    ))
    img = np.stack([b, g, r], -1)
    img = (img.clip(0, 1) * ddepth).astype(save_type)
    cv2.imwrite(save_path, img)

def save_rough_blender_exr(exr_file, save_path, ddepth=255):
    save_type = ddepth2dtype(ddepth)
    (img,) = read_channels(exr_file, ('ViewLayer.Roughness.B',))
    img = (img.clip(0, 1) * ddepth).astype(save_type)
    cv2.imwrite(save_path, img)

def save_material_mask_blender_exr(exr_file, save_path_dielec, save_path_metal, ddepth=255):
    save_type = ddepth2dtype(ddepth)
    img_dielec, img_metal = read_channels(
        exr_file, ('dielec.Combined.A', 'metal.Combined.A'),
    )
    mask_dielec = (((1-img_dielec)>0.5) * ddepth).astype(save_type)
    mask_metal = (((1-img_metal)>0.5) * ddepth).astype(save_type)
    cv2.imwrite(save_path_dielec, mask_dielec)
    cv2.imwrite(save_path_metal, mask_metal)

def optimize_bl_exr_save(exr_file, save_path):
    # Create an OpenEXR output file
    dw = exr_file.header()['dataWindow']
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
    
    preserved_keys = [
        'dielec.Combined.A', 'metal.Combined.A', # material segment mask
        'ViewLayer.DiffCol.B', 'ViewLayer.DiffCol.G', 'ViewLayer.DiffCol.R', # albedo
        'ViewLayer.Roughness.B', # roughness
        'ViewLayer.Normal.Z', 'ViewLayer.Normal.Y', 'ViewLayer.Normal.X', # normal
        'ViewLayer.Depth.Z', # depth
    ]
    exr_dict = {}
    for key in preserved_keys:
        exr_dict[key] = exr_file.channel(key, pt)

    # Write the data to the OpenEXR file
    exr_header = openexr.Header(size[1], size[0])
    exr_header['channels'] = {key: Imath.Channel(pt, 1, 1) for key in preserved_keys}
    output_exr = openexr.OutputFile(save_path, exr_header)
    output_exr.writePixels(exr_dict)
    output_exr.close()

def save_img_blender_exr(exr_file, save_path, ddepth=255):
    save_type = ddepth2dtype(ddepth)
    b, g, r = read_channels(exr_file, (
        'ViewLayer.Combined.B', 'ViewLayer.Combined.G', 'ViewLayer.Combined.R',
    ))
    img = np.stack([b, g, r], -1)
    img = (img.clip(0, 1)**(1./2.2) * ddepth).astype(save_type)
    cv2.imwrite(save_path, img)

def save_blender_results(save_prefix, info_dict_path, scene_dup_num=1):
    save_suffix = '_bl.exr'
    ddepth_a, suffix_a = 255, '_albedo_bl.png'
    ddepth_r, suffix_r = 255, '_rough_bl.png'
    ddepth_n, suffix_n = 65535, '_normal_bl.png'
    meta_info = np.load(info_dict_path, allow_pickle=True).item()
    
    cam_world_mats = meta_info['camera_transforms']
    scene_num = len(cam_world_mats)
    iters = scene_num * scene_dup_num
    for idx in range(iters):
        idx = iters - idx - 1
        src_idx = idx//scene_dup_num
        blender_exr_path = f'{save_prefix}_{idx+1:03d}{save_suffix}'
        if src_idx != idx:
            blender_exr_path_src = f'{save_prefix}_{src_idx+1:03d}{save_suffix}'
            if os.path.isfile(blender_exr_path_src):
                shutil.copy(blender_exr_path_src, blender_exr_path)
            else:
                raise FileNotFoundError(f'{blender_exr_path_src} not exsit')
        
        exr_file = openexr.InputFile(blender_exr_path)
        save_path_a = blender_exr_path.replace(save_suffix, suffix_a)
        save_albedo_blender_exr(exr_file, save_path_a, ddepth_a)
        save_path_r = blender_exr_path.replace(save_suffix, suffix_r)
        save_rough_blender_exr(exr_file, save_path_r, ddepth_r)
        save_path_n = blender_exr_path.replace(save_suffix, suffix_n)
        world_mat = cam_world_mats[src_idx]
        save_normal_blender_exr(exr_file, save_path_n, ddepth_n, world_mat, )

def save_blender_results_(save_prefix, cam_infos, scene_dup_num=1):
    save_suffix = '_bl.exr'
    ddepth_a, suffix_a = 255, '_albedo_bl.png'
    ddepth_r, suffix_r = 255, '_rough_bl.png'
    ddepth_n, suffix_n = 65535, '_normal_bl.png'
    suffix_dielec, suffix_metal = '_dielec_mask_bl.png', '_metal_mask_bl.png'
    
    scene_num = len(cam_infos)
    iters = scene_num * scene_dup_num
    for idx in range(iters):
        idx = iters - idx - 1
        src_idx = idx//scene_dup_num
        blender_exr_path = f'{save_prefix}_{idx+1:03d}{save_suffix}'
        if src_idx != idx:
            blender_exr_path_src = f'{save_prefix}_{src_idx+1:03d}{save_suffix}'
            if os.path.isfile(blender_exr_path_src):
                shutil.copy(blender_exr_path_src, blender_exr_path)
            else:
                raise FileNotFoundError(f'{blender_exr_path_src} not exsit')
        print(blender_exr_path)
        exr_file = openexr.InputFile(blender_exr_path)
        save_path_a = blender_exr_path.replace(save_suffix, suffix_a)
        save_albedo_blender_exr(exr_file, save_path_a, ddepth_a)
        save_path_r = blender_exr_path.replace(save_suffix, suffix_r)
        save_rough_blender_exr(exr_file, save_path_r, ddepth_r)
        save_path_n = blender_exr_path.replace(save_suffix, suffix_n)
        world_mat = cam_infos[src_idx]['transform']
        save_normal_blender_exr(exr_file, save_path_n, ddepth_n, world_mat, )
        save_path_dielec = blender_exr_path.replace(save_suffix, suffix_dielec)
        save_path_metal = blender_exr_path.replace(save_suffix, suffix_metal)
        save_material_mask_blender_exr(exr_file, save_path_dielec, save_path_metal)
        optimize_bl_exr_save(exr_file, blender_exr_path)
            

def save_mitsuba_results(save_prefix, chan_names, rendered_res=None):
    save_suffix = '_mi.exr'
    ddepth_i, suffix_i = 255, '_img_mi.jpg'
    ddepth_a, suffix_a = 255, '_albedo_mi.png'
    ddepth_n, suffix_n = 65535, '_normal_mi.png'
    suffix_pp = '_pprop_mi.jpg'
    # meta_save_path = f'{save_prefix}_meta.npy'
    rendered_save_path = f'{save_prefix}{save_suffix}'
    img_save_path = f'{save_prefix}{suffix_i}'
    albedo_save_path = f'{save_prefix}{suffix_a}'
    # normal_save_path = f'{save_prefix}{suffix_n}.png'
    polarprop_save_path = f'{save_prefix}{suffix_pp}'

    # cam_world_mat = np.array(mi.traverse(sensor)['to_world'].matrix)[0]
    if rendered_res is not None:
        rendered_bitmap = mi.Bitmap(rendered_res, channel_names=chan_names)
        rendered_bitmap.write(rendered_save_path)
    exr_file = openexr.InputFile(rendered_save_path)
    save_img_mitsuba_exr(exr_file, img_save_path, ddepth_i)
    # save_albedo_mitsuba_exr(exr_file, albedo_save_path, ddepth_a)
    # save_polarprop_exr(exr_file, polarprop_save_path)
    # save_normal_mitsuba_exr(rendered_save_path, normal_save_path, 65535, cam_world_mat)

def save_polar_mitsuba_exr(exr_file, mask, save_dir, save_name, ddepth=255, noise_std=None):
    save_type = ddepth2dtype(ddepth)
    s0b, s0g, s0r, s1b, s1g, s1r, s2b, s2g, s2r = read_channels(exr_file, (
        'S0.B', 'S0.G', 'S0.R',
        'S1.B', 'S1.G', 'S1.R',
        'S2.B', 'S2.G', 'S2.R',
    ))
    s0 = np.stack([s0b, s0g, s0r], axis=-1)
    s1 = np.stack([s1b, s1g, s1r], axis=-1)
    s2 = np.stack([s2b, s2g, s2r], axis=-1)
    stokes = np.stack([s0, s1, s2], axis=-1) # (h, w, c, s)
    # aolp = np.rad2deg(0.5 * np.arctan2(s2.mean(axis=-1), s1.mean(axis=-1)))
    # futil.visual_cmap(aolp, mask, aolp_cmap_info, 'check_scripts/results/polarprop_ori.png')
    
    pol000_ori = rutil.stks2polar(0, stokes) # (h, w, c)
    pol090_ori = rutil.stks2polar(90, stokes)
    pol045_ori = rutil.stks2polar(45, stokes)
    pol135_ori = rutil.stks2polar(135, stokes)
    
    pol = ((pol000_ori+pol090_ori+pol045_ori+pol135_ori)/4.).mean(axis=-1) # (h, w)
    
    mask = mask > 0.5 # (h, w)
    pol_mean = pol[mask].mean()
    
    dark_fg = False
    bright_fg = False
    scale = np.random.uniform(0.3, 0.4)
    num_try = 0
    while not (dark_fg*bright_fg):
        if scale < 0 or num_try > 20:
            return False
        pol000 = pol000_ori / pol_mean * scale
        pol090 = pol090_ori / pol_mean * scale
        pol045 = pol045_ori / pol_mean * scale
        pol135 = pol135_ori / pol_mean * scale

        pol_stack = np.concatenate([pol000,pol045,pol090,pol135], -1)
        pol_min = pol_stack.mean(axis=-1)
        pol_max = pol_stack.max(axis=-1)
        dark_mask = (pol_min < 0.00785) * mask
        exposed_mask = (pol_max > 0.996) * mask
        dark_ratio = dark_mask.sum()/mask.sum()
        bright_ratio = exposed_mask.sum()/mask.sum()
        print(f'\tScale: {scale:.5f}\tDark Region: {dark_ratio*100:.2f}% ' 
            f'Light Region: {bright_ratio*100:.2f}%')

        if dark_ratio + bright_ratio > 0.7:
            return False

        if not dark_fg:
            if dark_ratio > 0.40:
                scale += 0.15
                continue
            elif dark_ratio > 0.30:
                scale += 0.05
                continue
            else:
                dark_fg = True
        if not bright_fg:
            if bright_ratio > 0.1:
                scale -= 0.10
                continue
            if bright_ratio > 0.07:
                scale -= 0.06
                continue
            elif bright_ratio > 0.03:
                scale -= 0.02
                continue
            else:
                bright_fg = True

    if noise_std is not None:
        polar_imgs = np.stack([pol000, pol090, pol045, pol135], axis=0)
        polar_imgs = rutil.add_noise(polar_imgs, noise_std).clip(0, 1)
        # polar_imgs = rutil.gamma_corr(polar_imgs)
        pol000 = (polar_imgs[0]*ddepth).astype(save_type)
        pol090 = (polar_imgs[1]*ddepth).astype(save_type)
        pol045 = (polar_imgs[2]*ddepth).astype(save_type)
        pol135 = (polar_imgs[3]*ddepth).astype(save_type)
    else:
        pol000 = (pol000.clip(0, 1)*ddepth).astype(save_type)
        pol090 = (pol090.clip(0, 1)*ddepth).astype(save_type)
        pol045 = (pol045.clip(0, 1)*ddepth).astype(save_type)
        pol135 = (pol135.clip(0, 1)*ddepth).astype(save_type)
    # s1, s2 = pol000.astype(np.float32) - pol090.astype(np.float32), pol045.astype(np.float32) - pol135.astype(np.float32)
    # aolp = np.rad2deg(0.5*np.arctan2(s2.mean(axis=-1), s1.mean(axis=-1)))
    # futil.visual_cmap(aolp, mask.astype(np.float32), aolp_cmap_info, 'check_scripts/results/polarprop_img.png')
    # exit()

    pol000_path = os.path.join(save_dir, 'pol000', save_name)
    pol090_path = os.path.join(save_dir, 'pol090', save_name)
    pol045_path = os.path.join(save_dir, 'pol045', save_name)
    pol135_path = os.path.join(save_dir, 'pol135', save_name)
    mask_path = os.path.join(save_dir, 'mask', save_name)
    mask = (mask * 255).clip(0, 255).astype(np.uint8)

    img_write(pol000_path, pol000)
    img_write(pol090_path, pol090)
    img_write(pol045_path, pol045)
    img_write(pol135_path, pol135)
    img_write(mask_path, mask)
    return True


def visual_cmap(arr, mask, map_info, save_path):
    sizes = arr.shape[:2]
    dpi = max(sizes)
    fig = plt.figure()
    fig.set_size_inches(sizes[1] / sizes[0], 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(ax)
    ax.set_axis_off()
    ax.imshow(arr, alpha=mask, **map_info)
    plt.savefig(save_path, dpi=dpi, transparent=True) 
    ax.clear()
    plt.close()

def visual_cmap_(array, mask, map_info, save_path, save_fg=True):
    if len(mask.shape) == 2:
        mask = mask[..., None]
    cmap = matplotlib.colormaps[map_info['cmap']]
    vmin, vmax = map_info['vmin'], map_info['vmax']
    array = (array.clip(vmin, vmax) - vmin) / (vmax - vmin)
    rgba_map = cmap(array)
    rgb_map, a_map = rgba_map[..., :3]*mask, mask
    if save_fg:
        bgra_map = (np.concatenate([rgb_map[...,::-1], a_map], -1) * 255).astype(np.uint8)
        cv2.imwrite(save_path, bgra_map)
    return rgb_map

def save_summary_info(model_info, scene_info, save_path):
    info_dict = {**model_info, **scene_info}
    json_str = json.dumps(info_dict, indent=2, cls=JsonEncoder)
    with open(save_path, 'w') as f:
        f.write(json_str)