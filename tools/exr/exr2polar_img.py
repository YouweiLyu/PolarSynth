import os
import sys
import cv2
import Imath
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import utils.file_util as futil

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',  default='renderings/polar_event')
    parser.add_argument('--save_type',  default='png')
    parser.add_argument('-d', '--data_name',  default='')
    args = parser.parse_args()
    src_suffix = '.exr'
    dst_suffix = '.' + args.save_type
    img_save_dir = os.path.join(args.data_dir, args.save_type)
    # depth_save_dir = os.path.join(args.data_dir, 'depths')
    # normal_save_dir = os.path.join(args.data_dir, 'normals')
    save_dirs = [img_save_dir]
    for save_dir in save_dirs:
        os.makedirs(img_save_dir, exist_ok=True)

    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    if args.data_name:
        if args.data_name.endswith(src_suffix):
            file_path = os.path.join(args.data_dir, args.data_name)
        else:
            file_path = os.path.join(args.data_dir, args.data_name+src_suffix)
            args.data_name += src_suffix
        
        if os.path.isfile(file_path):
            print(file_path)
            futil.save_img_mitsuba_exrpath(file_path, os.path.join(img_save_dir, args.data_name.replace(src_suffix, dst_suffix)))
        else:
            raise NameError(f'{file_path} does not exist!')
    else:
        imgs = []
        save_paths = []
        for fn in sorted(os.listdir(args.data_dir)):
            if fn.endswith('.exr'):
                name = fn.replace('.exr', '')
                file_path = os.path.join(args.data_dir, fn)
                print(file_path)
                save_path = os.path.join(img_save_dir, name+'.png')
                save_paths.append(save_path)
                img = futil.save_img_mitsuba_exrpath(file_path, save_path)
                imgs.append(img)
            else:
                continue
    
        v_writer = cv2.VideoWriter(
            os.path.join(img_save_dir, 'demo.avi'),
            cv2.VideoWriter_fourcc(*'DIVX'),
            len(imgs)//3,
            (imgs[0].shape[1], imgs[0].shape[0])
        )
        with open(os.path.join(img_save_dir, 'info.txt'), 'w') as f:
            for i, img in enumerate(imgs):
                img_gamma = (((img/255) ** (1/2.2))*255).clip(0,255)
                v_writer.write(img_gamma.astype(np.uint8))
                f.write(f'{save_paths[i]} {int(1/len(imgs)*30000*(i+1)):012d}\n')
            v_writer.release()
