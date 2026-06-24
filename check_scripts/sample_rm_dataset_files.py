import os
import glob
import numpy as np

data_dir=os.path.join(os.environ.get('MITSUBA_DATA_DIR', './datasets'), 'training_set/sgl_obj_combined_copy')
subfolders = ['pol000', 'pol045', 'pol090', 'pol135', 'material_tag', 'normal', 'mask']

pol000_paths = glob.glob(f'{data_dir}/pol000/*_002.png')
if not pol000_paths:
    raise Exception()

num_delete = 0
for pol000_path in pol000_paths:
    name = os.path.basename(pol000_path)
    random_float = np.random.rand()
    if random_float < 0.4:
        continue
    else:
        num_delete += 1
        for folder in subfolders:
            delete_file_path = os.path.join(data_dir, folder, name)
            if os.path.isfile(delete_file_path):
                print(f'Delete: {delete_file_path}')
                os.remove(delete_file_path)
    
print(num_delete)
