import os
import numpy as np

data_dirs = [
    'assets/3D_assets/models/ambientcg',
    'assets/3D_assets/models/cgtrader',
    'assets/3D_assets/models/omniobject3D',
    'assets/3D_assets/models/polyhaven',
    # 'assets/models'
]

symmetric_txt_path = 'txts/model_is_symmetric.txt'
disp_prohibit_txt_path = 'txts/model_is_disp_prohibit.txt'
flat_txt_path = 'txts/model_is_flat.txt'

with open(symmetric_txt_path, 'r') as f:
    sym_model_paths = f.read().splitlines()
with open(flat_txt_path, 'r') as f:
    flat_model_paths = f.read().splitlines()
with open(disp_prohibit_txt_path, 'r') as f:
    disp_prohibit_paths = f.read().splitlines()

for data_dir in data_dirs:
    for fn in sorted(os.listdir(data_dir)):
        if fn.endswith('.obj'):
            print(fn)
            is_flat = False
            is_symmetric = False
            is_disp_prohibit = False
            filepath = os.path.join(data_dir, fn)
            file_meta_path = filepath.replace('.obj', '_objinfo.npy')

            x, y, z = [], [], []
            with open(filepath, 'r') as f:
                obj_lines = f.readlines()
            for line in obj_lines:
                if line.startswith('v '):
                    vertices = line.split(' ')[1:]
                    x.append(float(vertices[0]))
                    y.append(float(vertices[1]))
                    z.append(float(vertices[2]))
            vertex_num = len(x)
            x_center = (min(x) + max(x)) / 2
            y_center = (min(y) + max(y)) / 2
            z_center = (min(z) + max(z)) / 2

            if filepath in sym_model_paths:
                is_symmetric = True
            if filepath in flat_model_paths:
                is_flat = True
            if filepath in disp_prohibit_paths:
                is_disp_prohibit = True
        
            if os.path.isfile(file_meta_path) and file_meta_path.endswith('.npy'):
                obj_info = np.load(file_meta_path, allow_pickle=True).item()
            else:
                obj_info = {}
            
            obj_info['bbox_center'] = np.array([x_center, y_center, z_center])
            obj_info['vertex_num'] = vertex_num
            obj_info['is_symmetric'] = is_symmetric
            obj_info['is_flat'] = is_flat
            obj_info['is_disp_prohibit'] = is_disp_prohibit
            np.save(file_meta_path, obj_info)
            # print(x_center,y_center,z_center); exit()