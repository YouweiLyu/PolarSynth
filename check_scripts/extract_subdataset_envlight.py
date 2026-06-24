import os
import glob
import shutil
import random

ori_dir = os.path.join(os.environ.get('MITSUBA_DATA_DIR', './datasets'), 'training/sgl_obj_0922_3')
src_dir = os.path.join(os.environ.get('MITSUBA_DATA_DIR', './datasets'), 'training_set/sgl_obj_0922_3')
dst_dir = os.path.join(os.environ.get('MITSUBA_DATA_DIR', './datasets'), 'training_set/sgl_obj_0922_1234_40000_envlight_3')
copy_subdirs = ['pol000', 'pol045', 'pol090', 'pol135', 'mask', 'normal']
expect_num = 1720

if __name__ == '__main__':
    if os.path.isdir(dst_dir):
        print('dst dataset dir is existing')
    os.makedirs(dst_dir, exist_ok=True)
    for subdir in copy_subdirs:
        os.makedirs(os.path.join(dst_dir, subdir), exist_ok=True)
    
    pol000_paths = glob.glob(f'{src_dir}/pol000/*.png')
    random.shuffle(pol000_paths)
    extract_counter = 0
    for pol000_path in pol000_paths:
        otherlight_flag = False
        filename = os.path.basename(pol000_path).replace('.png', '')
        if os.path.isfile(os.path.join(dst_dir, 'mask', f'{filename}.png')):
            continue
        xml_path = os.path.join(ori_dir, filename+'.xml')
        if not os.path.isfile(xml_path):
            xml_path = os.path.join(ori_dir+'_2', filename+'.xml')
        with open(xml_path, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                if 'name="otherlight_1"' in line.strip():
                    otherlight_flag = True
                    break
        if otherlight_flag:
            continue
        for subdir in copy_subdirs:
            src_path = os.path.join(src_dir, subdir, f'{filename}.png')
            dst_path = os.path.join(dst_dir, subdir, f'{filename}.png')
            shutil.copy2(src_path, dst_path)

        extract_counter += 1
        print(f'{filename:<30s}[{extract_counter:>6d}/{expect_num:<6d}]')
        if extract_counter >= expect_num:
            break
    print(f'{src_dir} => {dst_dir}')
    print(f'### extracted: {extract_counter}; expected: {expect_num} ###')
