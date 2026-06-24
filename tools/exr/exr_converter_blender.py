import os
import sys
import OpenEXR
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import utils.file_util as futil

exr_dir = 'renderings_blender/polar_scenes'
meta_dir = 'renderings/polar_scenes/sgl_obj'

if __name__ == '__main__':
    suffix = 'blender.exr'
    for fn in os.listdir(os.path.join(exr_dir)):
        if fn.endswith(suffix):
            exr_path = os.path.join(exr_dir, fn)
            exr_file = OpenEXR.InputFile(exr_path)
            # print(exr_file.header()); exit()
            img_save_path = exr_path.replace('.exr', '_img.png')
            albedo_save_path = exr_path.replace('.exr', '_albedo.png')
            normal_save_path = exr_path.replace('.exr', '_normal.png')
            futil.save_img_blender_exr(exr_file, img_save_path)
            futil.save_albedo_blender_exr(exr_file, albedo_save_path)
            meta_path = os.path.join(meta_dir, fn.replace(suffix, 'meta.npy'))
            if os.path.exists(meta_path):
                meta_info = np.load(meta_path, allow_pickle=True).item()
                cam_world_mat = meta_info['camera_mat']
            else:
                cam_world_mat = None
            futil.save_normal_blender_exr(exr_file, normal_save_path, 65535, cam_world_mat)
