data_save_dir='renderings/polar_scenes_sgl_obj'
# python check_scripts/test_renderer_sgl_obj.py --debug \
#     --model_txt_path txts/models_sgl_obj_train_1.txt \
#     --cache_dir tmp/tmp_test --num_rendered_per_obj 2 --pose_num_per_scene 2 \
#     --save_dir $data_save_dir 
python render_sgl_obj.py --debug \
    --model_txt_path txts/models_sgl_obj_train.txt \
    --cache_dir tmp/tmp_1 --num_rendered_per_obj 2 --pose_num_per_scene 2 \
    --save_dir $data_save_dir