import os
import sys
import Imath
import OpenEXR as openexr
import numpy as np
import imageio
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import utils.polar_util as putil

data_dir = 'renderings'
save_dir = os.path.join(data_dir, 'polar_prop')

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

pt = Imath.PixelType(Imath.PixelType.FLOAT)

for fn in os.listdir(data_dir):
    if fn.endswith('_polar.exr'):
        name = fn.replace('_polar.exr', '')
        print(os.path.join(data_dir, fn))
    else:
        continue
    exr_file = openexr.InputFile(os.path.join(data_dir, fn))
    # print(exr_file.header()); exit()
    dw = exr_file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    s0_r = np.array(Image.frombytes("F", size, exr_file.channel("S0.R", pt)))
    s0_g = np.array(Image.frombytes("F", size, exr_file.channel("S0.G", pt)))
    s0_b = np.array(Image.frombytes("F", size, exr_file.channel("S0.B", pt)))
    s1_r = np.array(Image.frombytes("F", size, exr_file.channel("S1.R", pt)))
    s1_g = np.array(Image.frombytes("F", size, exr_file.channel("S1.G", pt)))
    s1_b = np.array(Image.frombytes("F", size, exr_file.channel("S1.B", pt)))
    s2_r = np.array(Image.frombytes("F", size, exr_file.channel("S2.R", pt)))
    s2_g = np.array(Image.frombytes("F", size, exr_file.channel("S2.G", pt)))
    s2_b = np.array(Image.frombytes("F", size, exr_file.channel("S2.B", pt)))
    
    # for R components
    r_dop, r_aop = putil.dop(s0_r,s1_r,s2_r), putil.aop(s1_r,s2_r)
    fig, ax = plt.subplots(1,3, figsize=(19,6), dpi=300)
    ax[0].axis('off')
    ax[0].set_title(r'Unpolarized image $\mathcal{R}$')
    ax_1 = ax[0].imshow(s0_r, cmap='gray', vmax=1, vmin=0)
    ax[1].axis('off')
    ax[1].set_title(r'DoP $\mathcal{R}$')
    ax_2 = ax[1].imshow(r_dop, cmap='GnBu', vmax=1, vmin=0)
    bar_1 = fig.colorbar(ax_2, ax=ax[1])
    ax[2].axis('off')
    ax[2].set_title(r'AoP $\mathcal{R}$')
    ax_3 = ax[2].imshow(np.rad2deg(r_aop), cmap='hsv', vmax=90, vmin=-90)
    bar_2 = fig.colorbar(ax_3, ax=ax[2])
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, name+'_polarInfo_R.jpg'))
    plt.close()
    # for G components
    g_dop, g_aop = putil.dop(s0_g,s1_g,s2_g), putil.aop(s1_g,s2_g)
    fig, ax = plt.subplots(1,3, figsize=(19,6), dpi=300)
    ax[0].axis('off')
    ax[0].set_title(r'Unpolarized image $\mathcal{G}$')
    ax_1 = ax[0].imshow(s0_g, cmap='gray', vmax=1, vmin=0)
    ax[1].axis('off')
    ax[1].set_title(r'DoP $\mathcal{G}$')
    ax_2 = ax[1].imshow(g_dop, cmap='GnBu', vmax=1, vmin=0)
    bar_1 = fig.colorbar(ax_2, ax=ax[1])
    ax[2].axis('off')
    ax[2].set_title(r'AoP $\mathcal{G}$')
    ax_3 = ax[2].imshow(np.rad2deg(g_aop), cmap='hsv', vmax=90, vmin=-90)
    bar_2 = fig.colorbar(ax_3, ax=ax[2])
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, name+'_polarInfo_G.jpg'))
    plt.close()
    # for B components
    b_dop, b_aop = putil.dop(s0_b,s1_b,s2_b), putil.aop(s1_b,s2_b)
    fig, ax = plt.subplots(1,3, figsize=(19,6), dpi=300)
    ax[0].axis('off')
    ax[0].set_title(r'Unpolarized image $\mathcal{B}$')
    ax_1 = ax[0].imshow(s0_b, cmap='gray', vmax=1, vmin=0)
    ax[1].axis('off')
    ax[1].set_title(r'DoP $\mathcal{B}$')
    ax_2 = ax[1].imshow(b_dop, cmap='GnBu', vmax=1, vmin=0)
    bar_1 = fig.colorbar(ax_2, ax=ax[1])
    ax[2].axis('off')
    ax[2].set_title(r'AoP $\mathcal{B}$')
    ax_3 = ax[2].imshow(np.rad2deg(b_aop), cmap='hsv', vmax=90, vmin=-90)
    bar_2 = fig.colorbar(ax_3, ax=ax[2])
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, name+'_polarInfo_B.jpg'))
    plt.close()
