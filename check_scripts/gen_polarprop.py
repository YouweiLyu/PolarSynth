import os
import glob
import numpy as np

import utils.file_util as futil

data_dir = '/home/lyuyouwei/UniSfP/data/test_set/sphere_casestudy_3'
aolp_map_info = {'cmap': 'hsv', 'vmin': -90, 'vmax': 90}
dolp_map_info = {'cmap': 'viridis', 'vmin': 0, 'vmax': 1}
stokes_map_info = {'cmap': 'RdBu', 'vmin': -0.5, 'vmax': 0.5}

if __name__ == '__main__':
    pol000_paths = glob.glob(f'{data_dir}/pol000/*.png')
    if not pol000_paths:
        raise FileNotFoundError()
    else:
        save_dir = os.path.join(data_dir, 'polarprop')
        os.makedirs(save_dir, exist_ok=True)
    for pol000_path in pol000_paths:
        fn = os.path.basename(pol000_path)
        pol045_path = pol000_path.replace('/pol000/', '/pol045/')
        pol090_path = pol000_path.replace('/pol000/', '/pol090/')
        pol135_path = pol000_path.replace('/pol000/', '/pol135/')
        mask_path = pol000_path.replace('/pol000/', '/mask/')
        mask = futil.img_read(mask_path)
        pol000, pol045 = futil.img_read(pol000_path).mean(axis=-1), futil.img_read(pol045_path).mean(axis=-1), 
        pol090, pol135 = futil.img_read(pol090_path).mean(axis=-1), futil.img_read(pol135_path).mean(axis=-1), 
        s0 = (pol000 + pol045 + pol090 + pol135) / 4
        s1 = (pol000 - pol090) / 2
        s2 = (pol045 - pol135) / 2
        eps = 1e-8
        aolp, dolp = 0.5*np.arctan2(s2, s1+eps), np.sqrt(s1**2+s2**2)/(s0+eps)
        
        aolp_path = os.path.join(save_dir, fn.replace('.png', '_aolp.png'))
        dolp_path = os.path.join(save_dir, fn.replace('.png', '_dolp.png'))
        s1_path = os.path.join(save_dir, fn.replace('.png', '_s1.png'))
        s2_path = os.path.join(save_dir, fn.replace('.png', '_s2.png'))
        futil.visual_cmap_(np.rad2deg(aolp), mask, aolp_map_info, aolp_path, True)
        futil.visual_cmap_(dolp, mask, dolp_map_info, dolp_path, True)
        futil.visual_cmap_(s1, mask, stokes_map_info, s1_path, True)
        futil.visual_cmap_(s2, mask, stokes_map_info, s2_path, True)