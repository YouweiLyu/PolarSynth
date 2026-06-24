import os

from check_scripts import MiConverter as convert

######## Hyperparameters ########
is_debug = False
is_shuffle = False
num_expected = 10000
src_data_txt = 'txts/sgl_obj_0922/envlight_noplane_dielectric_paths_train_4.txt'
src_dirs = [
    # os.path.join(os.environ.get('MITSUBA_DATA_DIR', './datasets'), 'test/sgl_obj_0922'),
    os.path.join(os.environ.get('MITSUBA_DATA_DIR', './datasets'), 'training/sgl_obj_0922_1_2'),
    os.path.join(os.environ.get('MITSUBA_DATA_DIR', './datasets'), 'training/sgl_obj_0922_2'),
    os.path.join(os.environ.get('MITSUBA_DATA_DIR', './datasets'), 'training/sgl_obj_0922_2_2'),
    os.path.join(os.environ.get('MITSUBA_DATA_DIR', './datasets'), 'training/sgl_obj_0922_3'),
    os.path.join(os.environ.get('MITSUBA_DATA_DIR', './datasets'), 'training/sgl_obj_0922_3_2'),
    os.path.join(os.environ.get('MITSUBA_DATA_DIR', './datasets'), 'training/sgl_obj_0922_4'),
    os.path.join(os.environ.get('MITSUBA_DATA_DIR', './datasets'), 'training/sgl_obj_0922_4_2'),
]
dst_dir = '/home/lyuyouwei/UniSfP/data/training_set/sgl_obj_0922_40k_envlight_dielectric'
convert_items = ['polar', 'normal', 'mask']
################################

if __name__ == '__main__':
    converter = convert.MiConverterEnvNoplaneDielec()
    converter.filter_mi_exr_paths(src_data_txt, src_dirs)
    converter.get_mi_exr_paths_from_txt(src_data_txt)
    converter.convert(convert_items, dst_dir, num_expected, is_debug, is_shuffle)
