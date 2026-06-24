import os
import cv2
import glob
import numpy as np
import OpenEXR as openexr

import utils.file_util as futil

data_dir = 'renderings/mult_4_obj'
exr_paths = glob.glob(f'{data_dir}/*.exr')
if not exr_paths:
    raise FileNotFoundError(f'{data_dir}')

for exr_path in exr_paths:
    exr_file = openexr.InputFile(exr_path)
    print(exr_path)
    print(exr_file.header())
    albedo_path = exr_path.replace('bl.exr', 'albedo_bl_save.png')
    futil.save_albedo_blender_exr(exr_file, albedo_path)
    rough_path = exr_path.replace('bl.exr', 'rough_bl_save.png')
    futil.save_rough_blender_exr(exr_file, rough_path)
    normal_path = exr_path.replace('bl.exr', 'normal_bl_save.png')
    futil.save_normal_blender_exr(exr_file, normal_path, save_suffix='save.png')
    dielec_path = exr_path.replace('bl.exr', 'dielec_mask_bl_save.png')
    metal_path = exr_path.replace('bl.exr', 'metal_mask_bl_save.png')
    futil.save_material_mask_blender_exr(exr_file, dielec_path, metal_path)

    
