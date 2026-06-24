import os
import cv2
import Imath
import argparse
import OpenEXR as openexr
import numpy as np
from PIL import Image

def save_exr2jpg(exr_path, save_path):
    exr_file = openexr.InputFile(exr_path)
    dw = exr_file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    r = np.array(Image.frombytes('F', size, exr_file.channel('R', pt)))
    g = np.array(Image.frombytes('F', size, exr_file.channel('G', pt)))
    b = np.array(Image.frombytes('F', size, exr_file.channel('B', pt)))
    img = np.stack([b, g, r], -1)
    img = (img * 255).clip(0, 255)
    cv2.imwrite(save_path, img)

def save_depth_mitsuba_exr(exr_path, save_path):
    exr_file = openexr.InputFile(exr_path)
    dw = exr_file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    depth = np.array(Image.frombytes('F', size, exr_file.channel('depth', pt)))
    d_max, d_min = depth[depth > 0].max(), depth[depth > 1e-6].min()
    img = (depth - d_min) / (d_max - d_min)
    img = (img * 255).clip(0, 255)
    cv2.imwrite(save_path, img)

def save_normal_mitsuba_exr(exr_path, save_path):
    exr_file = openexr.InputFile(exr_path)
    dw = exr_file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    normal_x = -np.array(Image.frombytes('F', size, exr_file.channel('normal.X', pt)))
    normal_y = np.array(Image.frombytes('F', size, exr_file.channel('normal.Y', pt)))
    normal_z = np.array(Image.frombytes('F', size, exr_file.channel('normal.Z', pt)))
    normals = np.stack([normal_y, normal_z, normal_x], -1) 
    normal_mod = (normals ** 2).sum(axis=-1, keepdims=True)
    depth = np.array(Image.frombytes('F', size, exr_file.channel('depth', pt)))
    mask = (depth[..., None] > 0) * (normal_mod > 0.1)
    normals = normals / (normal_mod+1e-6) * mask

    normals = ((normals + 1) / 2).clip(0, 1) * 65535
    cv2.imwrite(save_path, normals.astype(np.uint16))
    save_path = save_path.replace('.png', '_mask.png')
    cv2.imwrite(save_path, mask*255)
    save_path = save_path.replace('_mask.png', '_norm.png')
    cv2.imwrite(save_path, normal_mod*255)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',  default='renderings')
    parser.add_argument('-d', '--data_name',  default='')
    args = parser.parse_args()
    img_save_dir = os.path.join(args.data_dir, 'jpg')
    depth_save_dir = os.path.join(args.data_dir, 'depths')
    normal_save_dir = os.path.join(args.data_dir, 'normals')
    save_dirs = [img_save_dir, depth_save_dir, normal_save_dir]
    for save_dir in save_dirs:
        if not os.path.isdir(img_save_dir):
            os.makedirs(img_save_dir)

    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    if args.data_name:
        if args.data_name.endswith('.exr'):
            file_path = os.path.join(args.data_dir, args.data_name)
        else:
            file_path = os.path.join(args.data_dir, args.data_name+'.exr')
            args.data_name += '.exr'
        
        if os.path.isfile(file_path):
            print(file_path)
            save_exr2jpg(file_path, os.path.join(img_save_dir, args.data_name.replace('.exr', '.jpg')))
            save_normal_mitsuba_exr(file_path, os.path.join(normal_save_dir, args.data_name.replace('.exr', '.png')))
            # save_depth_mitsuba_exr(file_path, os.path.join(depth_save_dir, args.data_name.replace('.exr', '.jpg')))

        else:
            raise NameError(f'{file_path} does not exist!')
    else:
        for fn in os.listdir(args.data_dir):
            if fn.endswith('.exr'):
                name = fn.replace('.exr', '')
                file_path = os.path.join(args.data_dir, fn)
                print(file_path)
                save_path = os.path.join(img_save_dir, name+'.jpg')
                save_exr2jpg(file_path, save_path)
                save_path = os.path.join(normal_save_dir, name+'.png')
                save_normal_mitsuba_exr(file_path, save_path)
                # save_path = os.path.join(depth_save_dir, name+'.jpg')
                # save_depth_mitsuba_exr(file_path, save_path)
            else:
                continue