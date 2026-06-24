import glob
import os
import shutil

dst_dir = 'assets/3D_assets/models/objaverse_sketchfab_obj'
txt_path = 'obj_files.txt'

with open(txt_path, 'r') as f:
    paths = f.read().splitlines()
    if not paths:
        raise Exception()
    
    os.makedirs(dst_dir, exist_ok=True)
    for path in paths:
        obj_path = os.path.join(path.replace('.glb', ''), 'scene.obj')
        dst_path = os.path.join(dst_dir, os.path.basename(path).replace('.glb', '.obj'))
        shutil.copy(obj_path, dst_path)
