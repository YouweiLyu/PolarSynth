gpu_id=$1
mi_variant=scalar_spectral_polarized
training_data_dir="${MITSUBA_DATA_DIR:-./datasets}/training/sgl_obj_0922"
test_data_dir="${MITSUBA_DATA_DIR:-./datasets}/test/sgl_obj_1206"
save_dir='renderings/test_2'
export CUDA_VISIBLE_DEVICES=$gpu_id
export MI_DEFAULT_VARIANT=$mi_variant
{
# python render_case_study.py --mi_variant $mi_variant \
#     --cache_dir tmp/tmp_test --save_dir $save_dir \
#     --pose_num_per_scene 1  --workers 16
# python render_case_study.py --mi_variant $mi_variant \
#     --model_txt_path txts/models_sgl_obj_test_.txt --cache_dir tmp/tmp_1 --save_dir ${tmp_dir} \
#     --material_txt_path txts/materials_sgl_obj_test.txt --hdri_txt_path txts/hdri_sgl_obj_test.txt\
#     --num_rendered_per_obj 50 --pose_num_per_scene 1 --start_model_index 0 --workers 8
python render_sgl_obj.py --mi_variant $mi_variant \
    --model_txt_path txts/models_sgl_obj_test.txt --material_txt_path txts/materials_sgl_obj_test.txt --hdri_txt_path txts/hdri_sgl_obj_test.txt \
    --cache_dir tmp/tmp_test --num_rendered_per_obj 10 --pose_num_per_scene 8 --save_dir $save_dir --workers 16
}