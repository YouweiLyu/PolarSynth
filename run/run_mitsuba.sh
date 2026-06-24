mi_variant=scalar_spectral_polarized
export MI_DEFAULT_VARIANT=$mi_variant
mitsuba -m $mi_variant -t 16 -o test.exr 'scenes/sphere_polar_materials.xml' 
python exr_converter_mitsuba.py