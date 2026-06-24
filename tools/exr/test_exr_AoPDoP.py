import os
import sys
import Imath
import OpenEXR as openexr
import numpy as np
import argparse
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import utils.polar_util as putil

def save_polar_info(exr_path, save_path):
    exr_file = openexr.InputFile(exr_path)
    # print(exr_file.header()); exit()
    dw = exr_file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    s0_r = np.array(Image.frombytes("F", size, exr_file.channel("S0.R", pt)))
    s0_g = np.array(Image.frombytes("F", size, exr_file.channel("S0.G", pt)))
    s0_b = np.array(Image.frombytes("F", size, exr_file.channel("S0.B", pt)))
    s0 = (s0_r + s0_g + s0_b) / 3
    s1_r = np.array(Image.frombytes("F", size, exr_file.channel("S1.R", pt)))
    s1_g = np.array(Image.frombytes("F", size, exr_file.channel("S1.G", pt)))
    s1_b = np.array(Image.frombytes("F", size, exr_file.channel("S1.B", pt)))
    s1 = (s1_r + s1_g + s1_b) / 3
    s2_r = np.array(Image.frombytes("F", size, exr_file.channel("S2.R", pt)))
    s2_g = np.array(Image.frombytes("F", size, exr_file.channel("S2.G", pt)))
    s2_b = np.array(Image.frombytes("F", size, exr_file.channel("S2.B", pt)))
    s2 = (s2_r + s2_g + s2_b) / 3
    
    r_dop, r_aop = putil.dop(s0,s1,s2), putil.aop(s1,s2)
    fig, ax = plt.subplots(1,3, figsize=(19,6), dpi=300)
    ax[0].axis('off')
    ax[0].set_title(r'Unpolarized image')
    ax_1 = ax[0].imshow(s0, cmap='gray', vmax=1, vmin=0)
    ax[1].axis('off')
    ax[1].set_title(r'DoP')
    ax_2 = ax[1].imshow(r_dop, cmap='GnBu', vmax=1, vmin=0)
    bar_1 = fig.colorbar(ax_2, ax=ax[1])
    ax[2].axis('off')
    ax[2].set_title(r'AoP')
    ax_3 = ax[2].imshow(np.rad2deg(r_aop), cmap='hsv', vmax=90, vmin=-90)
    bar_2 = fig.colorbar(ax_3, ax=ax[2])
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',  default='renderings')
    parser.add_argument('-d', '--data_name',  default='')
    args = parser.parse_args()
    save_dir = os.path.join(args.data_dir, 'polar_prop')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    if args.data_name:
        if args.data_name.endswith('.exr'):
            file_path = os.path.join(args.data_dir, args.data_name)
        else:
            file_path = os.path.join(args.data_dir, args.data_name+'.exr')
            args.data_name += '.exr'
        
        if os.path.isfile(file_path):
            print(file_path)
            save_polar_info(file_path, os.path.join(save_dir, args.data_name.replace('.exr', '.jpg')))
        else:
            raise NameError(f'{file_path} does not exist!')
    else:
        for fn in os.listdir(args.data_dir):
            if fn.endswith('_polar.exr'):
                name = fn.replace('_polar.exr', '')
                file_path = os.path.join(args.data_dir, fn)
                print(file_path)
                save_path = os.path.join(save_dir, name+'_polarInfo.jpg')
                save_polar_info(file_path, save_path)
            else:
                continue
            
            