import os

def check_file_exist(filepaths):
    missing_files = []
    for fp in filepaths:
        if not os.path.isfile(fp):
            missing_files.append(fp)

    if missing_files:
        print()
        print(os.path.split(missing_files[0])[0])
        for fpath in missing_files:
            print(os.path.split(fpath)[-1])
            if 'Color' in fpath and os.path.isfile(fpath.replace('Color', 'Colorq')):
                os.rename(fpath.replace('Color', 'Colorq'), fpath)


res = '4K'
format = 'jpg'
material_txt_path = 'txts/materials_sgl_obj_test.txt'
with open (material_txt_path, 'r') as f:
    material_dirs = f.read().splitlines()
for m_dir in material_dirs:
    if len(os.listdir(m_dir)) < 1:
        print('empty dir:', m_dir)
    for m_subdir in sorted(os.listdir(m_dir)):
        material_dir = os.path.join(m_dir, m_subdir)
        if os.path.isdir(material_dir):
            m_name = os.path.split(material_dir)[-1]
            displace_path = os.path.join(material_dir, f'{m_name}_{res.upper()}_Displacement.{format}')
            normal_path = os.path.join(material_dir, f'{m_name}_{res.upper()}_NormalGL.{format}')
            color_path = os.path.join(material_dir, f'{m_name}_{res.upper()}_Color.{format}')
            rough_path = os.path.join(material_dir, f'{m_name}_{res.upper()}_Roughness.{format}')
            filepaths = [displace_path,normal_path,color_path,rough_path]
            check_file_exist(filepaths)

