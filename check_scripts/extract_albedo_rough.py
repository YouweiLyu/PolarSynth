import os
import utils.file_util as futil
import OpenEXR as openexr

exr_paths = [
    # 'renderings/check_renderings2/sphere_ori_bl.exr',
    'renderings/check_renderings2/sphere_dup_bl.exr',
]
for exr_path in exr_paths:
    exr_file = openexr.InputFile(exr_path)
    print(exr_file.header())
    save_dir = os.path.dirname(exr_path)
    save_name = os.path.basename(exr_path).split('.')[0]+'_albedo.png'
    futil.save_albedo_blender_exr(exr_file, os.path.join(save_dir, save_name))