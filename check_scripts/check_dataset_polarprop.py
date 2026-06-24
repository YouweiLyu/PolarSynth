import os
import cv2
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')

dataset_dir = os.path.join(os.environ.get('MITSUBA_DATA_DIR', './datasets'), 'test_set/mult_objs_1213_1500_scenes')
save_folder = 'polarprop_img'


if __name__ == '__main__':
    save_dir = os.path.join(dataset_dir, save_folder)
    os.makedirs(save_dir, exist_ok=True)
    pol000_dir = os.path.join(dataset_dir, 'pol000')
    pol000_paths = glob.glob(f"{pol000_dir}/*.png")
    for pol000_path in pol000_paths:
        filename = os.path.basename(pol000_path)
        print(filename)
        pol000_img = (cv2.imread(pol000_path, -1).astype(np.float32) / 255.).mean(axis=-1)
        pol090_img = (cv2.imread(pol000_path.replace('pol000/', 'pol090/'), -1).astype(np.float32) / 255.).mean(axis=-1)
        pol045_img = (cv2.imread(pol000_path.replace('pol000/', 'pol045/'), -1).astype(np.float32) / 255.).mean(axis=-1)
        pol135_img = (cv2.imread(pol000_path.replace('pol000/', 'pol135/'), -1).astype(np.float32) / 255.).mean(axis=-1)
        s0 = (pol000_img + pol090_img + pol045_img + pol135_img) / 2.
        s1 = pol000_img - pol090_img
        s2 = pol045_img - pol135_img

        aolp, dolp = 0.5*np.arctan2(s2, s1+1e-6), np.sqrt(s1**2+s2**2)/(s0+1e-6)
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
        fig.savefig(os.path.join(save_dir, filename))
        plt.close('all')
        # gc.collect()


