import os 
import numpy as np
from PIL import Image

materia_txt_path = 'txts/materials_sgl_obj_test.txt'

with open(materia_txt_path, 'r') as f:
    material_dirs = f.read().splitlines()
for m_dir in material_dirs:
    if len(os.listdir(m_dir)) < 1:
        print('empty dir: ', m_dir)
    for m_subdir in os.listdir(m_dir):
        material_dir = os.path.join(m_dir, m_subdir)
        print(material_dir)
        m_name = os.path.split(material_dir)[-1]
        base_color_path = os.path.join(material_dir, f'{m_name}_baseColor.tga')
        diffuse_path = os.path.join(material_dir, f'{m_name}_diffuse.tga')
        roughness_path = os.path.join(material_dir, f'{m_name}_roughness.tga')
        displacement_path = os.path.join(material_dir, f'{m_name}_height.png')
        normal_path = os.path.join(material_dir, f'{m_name}_normal.png')

        base_color = Image.open(base_color_path).convert('RGB')
        base_color.save(os.path.join(material_dir, f'{m_name}_4K_Color.jpg'))
        # diffuse = Image.open(diffuse_path).convert('RGB')
        # diffuse.save(f'{m_name}_diffuse.png')
        roughness = Image.open(roughness_path).convert('RGB')
        roughness.save(os.path.join(material_dir, f'{m_name}_4K_Roughness.jpg'))
        displace = Image.open(displacement_path).convert('RGB')
        displace.save(os.path.join(material_dir, f'{m_name}_4K_Displacement.jpg'))
        normal = Image.open(normal_path).convert('RGB')
        normal.save(os.path.join(material_dir, f'{m_name}_4K_NormalGL.jpg'))