import os
import cv2
import utils.sampler_util as sutil

materials_txt_dir = 'txts/materials_sgl_obj_train.txt'
max_size = int(0x7fff)
materials_dirs = sutil.load_path_text(materials_txt_dir)
for materials_dir in materials_dirs:
    for material_folder in os.listdir(materials_dir):
        material_file_dir = os.path.join(materials_dir, material_folder)
        if os.path.isdir(material_file_dir):
            # print(material_file_dir)
            for file in os.listdir(material_file_dir):
                filepath = os.path.join(material_file_dir, file)
                if file.endswith('.jpg'):
                    img = cv2.imread(filepath)
                    size_max = max(img.shape)
                    if size_max > max_size:
                        print(material_file_dir)
                    else:
                        pass
                    break
                
