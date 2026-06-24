from check_scripts import MiConverter as converter

"""
Convert Mitsuba Rendered *.exr file to images.
Required: 
    Mitsuba Files: *_mi.exr
    Mask from Blender: *_mask_bl.png
"""

######## Hyperparameters ########
is_debug = False
is_shuffle = False
num_expected = 1000
src_dir = 'renderings/check_model_integrity/testset231213_2nd_exr'
# dst_dir = os.path.join(os.environ.get('MITSUBA_DATA_DIR', './datasets'), 'training_set/sgl_obj_1213_10k')
dst_dir = 'renderings/check_model_integrity/testset231213_2nd_converted'
convert_items = [
    'polar', 'normal', 'mask', # 'material_tag', 'albedo', 'roughness', 'normal_gl'
]
################################

if __name__ == '__main__':
    converter_base = converter.MiConverterBase()
    converter_base.get_mi_exr_paths_from_dir(src_dir)
    converter_base.convert(convert_items, dst_dir, num_expected, is_debug, is_shuffle)
    print(src_dir, '=>', dst_dir)
