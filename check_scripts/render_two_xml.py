
import utils.file_util as futil
import subprocess
import os

xml_paths  = [
    'renderings/polar_scenes_sgl_obj/Pastry2_001_001_.xml',
    'renderings/polar_scenes_sgl_obj/Pastry2_001_001.xml'
]
for idx, xml_path in enumerate(xml_paths):
    save_prefix = os.path.join('check_scripts/images', str(idx))
    subprocess.run(['mitsuba', '-m', 'llvm_spectral_polarized', '-o', f'{save_prefix}_mi.exr', xml_path], 
                    check=True, stdout=subprocess.DEVNULL)
    aov_names = ['albedo.R', 'albedo.G', 'albedo.B', 'stokes.S0.R', 'stokes.S0.G', 'stokes.S0.B',
                 'stokes.S1.R', 'stokes.S1.G', 'stokes.S1.B', 'stokes.S2.R', 'stokes.S2.G', 'stokes.S2.B', ]
    chan_names = ['R','G','B'] + aov_names
    futil.save_mitsuba_results(save_prefix, chan_names)