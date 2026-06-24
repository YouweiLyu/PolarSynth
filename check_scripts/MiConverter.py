import os
import cv2
import glob
import OpenEXR as openexr
import shutil
import random
import numpy as np

import utils.file_util as futil

class MiConverterBase:
    def __init__(self):
        self.img_noise_std_range = (0.002, 0.004)
        self.polar_ddepth, self.albedo_ddepth, self.rough_ddepth = 255, 255, 255
        self.mi_exr_suffix, self.bl_exr_suffix, self.bl_rough_suffix = '_mi.exr', '_bl.exr', '_rough_bl.exr'
        self.normal_lc_suffix, self.normal_gl_suffix = '_normal_L_bl.png', '_normal_G_bl.png'
        self.albedo_suffix, self.rough_suffix = '_albedo_bl.png', '_rough_bl.png'
        self.mask_suffix = '_mask_bl.png'
        self.xml_suffix = '.xml'
        self.dielec_suffix, self.metal_suffix = '_dielec_mask_bl.png', '_metal_mask_bl.png'

    def __check_mi_exr_paths(self):
        if not self.mi_exr_paths:
            raise FileNotFoundError(f'No expected EXR file found')

    def init_mi_exr_paths(self, mi_exr_paths):
        self.mi_exr_paths = mi_exr_paths
        self.__check_mi_exr_paths()

    def get_mi_exr_paths_from_dir(self, src_dir):
        self.mi_exr_paths = glob.glob(f"{src_dir}/*{self.mi_exr_suffix}")
        self.__check_mi_exr_paths()

    def get_mi_exr_paths_from_txt(self, text_path):
        with open(text_path, 'r') as f:
            self.mi_exr_paths = f.read().splitlines()
        self.__check_mi_exr_paths()

    def gather_mi_exr_paths_from_dirs(self, src_dirs):
        mi_exr_paths = []
        for src_dir in src_dirs:
            src_dir_mi_exr_paths = glob.glob(f"{src_dir}/*{self.mi_exr_suffix}")
            if not src_dir_mi_exr_paths:
                raise FileNotFoundError(f'No expected EXR file in {src_dir}')
            
            mi_exr_paths += src_dir_mi_exr_paths

        mi_exr_paths.sort()
        return mi_exr_paths

    def convert(self, convert_items, dst_dir, num_expected, is_debug=False, is_shuffle=False):
        if not self.mi_exr_paths:
            raise FileNotFoundError(f'No expected EXR file found')
        
        if is_debug:
            print('######## Debug Mode ########')
            print('Save polarprop for checking ...')
        
        if is_shuffle:
            random.shuffle(self.mi_exr_paths)
        else:
            self.mi_exr_paths.sort()
        
        futil.make_save_dirs(dst_dir, convert_items)
        
        num_converted = 0
        for mi_exr_path in self.mi_exr_paths:
            name = os.path.basename(mi_exr_path).replace(self.mi_exr_suffix, '')
            print(f'{name:<30s}[{num_converted:>6d}/{num_expected:<6d}]')
            mask_src_path = mi_exr_path.replace(self.mi_exr_suffix, self.mask_suffix)
            if 'polar' in convert_items:
                mi_exr_file = openexr.InputFile(mi_exr_path)
                std = np.random.uniform(self.img_noise_std_range[0], self.img_noise_std_range[1])
                mask = futil.img_read(mask_src_path)
                is_saved = futil.save_polar_mitsuba_exr(mi_exr_file, mask, dst_dir, name+'.png', self.polar_ddepth, std)
                if not is_saved:
                    print(f'fail to save "{mi_exr_path}"')
                    continue

                if is_debug:
                    futil.save_polarprop_exr(mi_exr_file, os.path.join(dst_dir, 'polarprop_exr', name+'.png'))
                
            if 'normal' in convert_items:
                normal_lc_src_path = mi_exr_path.replace(self.mi_exr_suffix, self.normal_lc_suffix)
                normal_lc_dst_path = os.path.join(dst_dir, 'normal', name+'.png')
                shutil.copy(normal_lc_src_path, normal_lc_dst_path)

            if 'normal_gl' in convert_items:
                normal_gl_src_path = mi_exr_path.replace(self.mi_exr_suffix, self.normal_gl_suffix)
                normal_gl_dst_path = os.path.join(dst_dir, 'normal_gl', name+'.png')
                shutil.copy(normal_gl_src_path, normal_gl_dst_path)

            if 'albedo' in convert_items:
                # bl_exr_path = mi_exr_path.replace(mi_exr_suffix, bl_exr_suffix)
                # bl_exr_file = openexr.InputFile(bl_exr_path)
                # albedo_dst_path = os.path.join(dst_dir, 'albedo', name+'.png')
                # futil.save_albedo_blender_exr(bl_exr_file, albedo_dst_path, albedo_ddepth)
                albedo_src_path = mi_exr_path.replace(self.mi_exr_suffix, self.albedo_suffix)
                albedo_dst_path = os.path.join(dst_dir, 'albedo', name+'.png')
                shutil.copy(albedo_src_path, albedo_dst_path)

            if 'roughness' in convert_items:
                # bl_exr_path = mi_exr_path.replace(mi_exr_suffix, bl_rough_suffix)
                # rough_dst_path = os.path.join(dst_dir, 'roughness', name+'.png')
                # if not os.path.isfile(bl_exr_path):
                #     cv2.imwrite(rough_dst_path, np.zeros((512, 512)))
                # else:
                #     bl_exr_file = openexr.InputFile(bl_exr_path)
                #     futil.save_rough_blender_exr(bl_exr_file, rough_dst_path, rough_ddepth)
                rough_src_path = mi_exr_path.replace(self.mi_exr_suffix, self.rough_suffix)
                rough_dst_path = os.path.join(dst_dir, 'roughness', name+'.png')
                shutil.copy(rough_src_path, rough_dst_path)

            if 'material_tag' in convert_items:
                tag_dielec_src_path = mi_exr_path.replace(self.mi_exr_suffix, self.dielec_suffix)
                print(tag_dielec_src_path)
                tag_metal_src_path = mi_exr_path.replace(self.mi_exr_suffix, self.metal_suffix)
                tag_dielec = cv2.imread(tag_dielec_src_path, -1).astype(np.uint8) / 255.
                tag_metal = cv2.imread(tag_metal_src_path, -1).astype(np.uint8) / 255.
                tag_dielec, tag_metal = tag_dielec > 0.5, tag_metal > 0.5
                pad = tag_dielec + tag_metal
                tag_img = np.stack([pad*120, tag_metal*255, tag_dielec*255], -1).astype(np.uint8)
                tag_dst_path = os.path.join(dst_dir, 'material_tag', name+'.png')
                cv2.imwrite(tag_dst_path, tag_img)
            
            num_converted += 1
            
            if is_debug and num_converted > 2:
                break

            if num_converted >= num_expected:
                break

        print(f'### Converted: {num_converted}; Expected: {num_expected} ###')

class MiConverterEnvNoplane(MiConverterBase):
    def __init__(self):
        super().__init__()
        
    def filter_mi_exr_paths(self, txt_path, src_dirs):
        if not os.path.isfile(txt_path):
            exr_paths_filtered = []
            print(f'Create data path txt => {txt_path}')
            os.makedirs(os.path.dirname(txt_path), exist_ok=True)
            mi_exr_paths = self.gather_mi_exr_paths_from_dirs(src_dirs)
            for mi_exr_path in mi_exr_paths:
                xml_path = mi_exr_path.replace(self.mi_exr_suffix, self.xml_suffix)
                fg = False
                with open(xml_path, 'r') as f:
                    lines = f.read().splitlines()
                    for line in lines:
                        if 'name="otherlight_1"' in line.strip() or 'id="model_1"' in line.strip():
                            fg = True
                            break
                if fg: 
                    continue
                exr_paths_filtered.append(f'{mi_exr_path}\n')
            
            if exr_paths_filtered:
                with open(txt_path, 'w') as f:
                    f.write(''.join(exr_paths_filtered))

class MiConverterEnvNoplaneDielec(MiConverterBase):
    def __init__(self):
        super().__init__()
        
    def filter_mi_exr_paths(self, txt_path, src_dirs):
        if not os.path.isfile(txt_path):
            exr_paths_filtered = []
            print(f'Create data path txt => {txt_path}')
            os.makedirs(os.path.dirname(txt_path), exist_ok=True)
            mi_exr_paths = self.gather_mi_exr_paths_from_dirs(src_dirs)
            for mi_exr_path in mi_exr_paths:
                xml_path = mi_exr_path.replace(self.mi_exr_suffix, self.xml_suffix)
                fg = False
                with open(xml_path, 'r') as f:
                    lines = f.read().splitlines()
                    for line in lines:
                        if 'conductor' in line.strip() or 'name="otherlight_1"' in line.strip() or 'id="model_1"' in line.strip():
                            fg = True
                            break
                if fg: 
                    continue
                exr_paths_filtered.append(f'{mi_exr_path}\n')
            
            if exr_paths_filtered:
                with open(txt_path, 'w') as f:
                    f.write(''.join(exr_paths_filtered))
