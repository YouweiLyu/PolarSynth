import os
import cv2
import glob
import shutil

import utils.file_util as futil

data_dir = os.path.join(os.environ.get('MITSUBA_DATA_DIR', './datasets'), 'test_set/mult_objs_1213_1500_scenes')
subfolders = ['mask', 'pol000', 'pol045', 'pol090', 'pol135', 'normal']

if __name__ == '__main__':
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(data_dir)

    save_dir = data_dir + '_dielec'
    shutil.copytree(data_dir, save_dir)
    material_paths = glob.glob(f'{save_dir}/material_tag/*.png')
    for m_path in material_paths:
        print(m_path)
        m_img = futil.img_read(m_path)
        dielec_mask = m_img[..., 2:3] > 0.8
        if dielec_mask.sum() < 100:
            print('delete')
            os.remove(m_path)
            for subf in subfolders:
                img_path = m_path.replace('/material_tag', f'/{subf}')
                os.remove(img_path)
            
        else:
            print('mask')
            m_img *= dielec_mask
            cv2.imwrite(m_path, m_img*255)
            for subf in subfolders:
                img_path = m_path.replace('/material_tag', f'/{subf}')
                img = cv2.imread(img_path, -1)
                if len(img.shape) == 2:
                    img_n = img * dielec_mask[..., 0]
                else:
                    img_n = img * dielec_mask

                cv2.imwrite(img_path, img_n)



