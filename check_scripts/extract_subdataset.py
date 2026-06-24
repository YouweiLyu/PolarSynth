import os
import glob
import shutil
import random

src_dir = os.path.join(os.environ.get('MITSUBA_DATA_DIR', './datasets'), 'test_set/sgl_obj_0922_6000')
dst_dir = os.path.join(os.environ.get('MITSUBA_DATA_DIR', './datasets'), 'test_set/sgl_obj_0922_12_1000')
copy_subdirs = ['pol000', 'pol045', 'pol090', 'pol135', 'mask', 'normal']
expect_num = 1000

if __name__ == '__main__':
    if os.path.isdir(dst_dir):
        raise Exception('dst dataset dir is existing')
    os.makedirs(dst_dir)
    for subdir in copy_subdirs:
        os.makedirs(os.path.join(dst_dir, subdir))
    
    pol000_paths = glob.glob(f'{src_dir}/pol000/*.png')
    random.shuffle(pol000_paths)
    extract_counter = 0
    for pol000_path in pol000_paths:
        filename = os.path.basename(pol000_path).replace('.png', '')
        for subdir in copy_subdirs:
            src_path = os.path.join(src_dir, subdir, f'{filename}.png')
            dst_path = os.path.join(dst_dir, subdir, f'{filename}.png')
            shutil.copy2(src_path, dst_path)

        extract_counter += 1
        if extract_counter >= expect_num:
            break
    print(f'{src_dir} => {dst_dir}')
    print(f'### extracted: {extract_counter}; expected: {expect_num} ###')
