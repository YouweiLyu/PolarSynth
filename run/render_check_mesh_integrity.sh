gpu_id=0
mi_variant=llvm_spectral_polarized
export CUDA_VISIBLE_DEVICES=$gpu_id
export MI_DEFAULT_VARIANT=$mi_variant
# Note: this python script does not produce material tags.
{
python check_scripts/render_check_mesh_integrity.py --mi_variant $mi_variant \
    --model_txt_path txts/models_mult_obj_test_2nd_check.txt \
    --save_dir 'renderings/check_model_integrity/testset231213_2nd_exr' --workers 16
exit
}