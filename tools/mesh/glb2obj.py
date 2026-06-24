import bpy
import glob
import json
import numpy as np
import sys
import os

def load_pplastic_model(
        albedo, roughness, model_path=None, normal_variant=None, 
        ior=1.4, dist='beckmann', model_type='sphere', 
        transform=None):
    model_dict = {'type': model_type,}
    bsdf_dict = {
        'type': 'pplastic',
        'diffuse_reflectance':{
            'type': 'bitmap',
            'wrap_mode': 'repeat',
        },
        'int_ior': ior,
        'distribution': dist,
        'alpha': {
            'type': 'bitmap',
            'wrap_mode': 'repeat',
        },
    }
    if isinstance(albedo, str):
        bsdf_dict['diffuse_reflectance']['filename'] = albedo
    elif isinstance(albedo, float):
        bsdf_dict['diffuse_reflectance'] = albedo
    else:
        raise TypeError(f'Invalid type {type(albedo)} for albedo')
    
    if isinstance(roughness, str):
        bsdf_dict['alpha']['filename'] = roughness
    elif isinstance(roughness, float):
        bsdf_dict['alpha'] = roughness
    else:
        raise TypeError(f'Invalid type {type(roughness)} for roughness')
    
    if model_path is not None:
        model_dict['filename'] = model_path
    if normal_variant is not None:
        model_dict['normalbsdf'] = {
            'type': 'normalmap',
            'normalmap': {
                'type': 'bitmap',
                'raw': True,
                'bitmap': normal_variant,
            },
            'nested_bsdf': bsdf_dict,
        }
    else:
        model_dict['bsdf'] = bsdf_dict
    if transform is not None:
        model_dict['to_world'] = transform
    
    return model_dict

def main():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]

    # Load model
    glb_dir = argv[0]
    if glb_dir.endswith('/'):
        glb_dir = glb_dir[:-1]

    glb_paths = glob.glob(f'{glb_dir}/*.glb')
    if not glb_paths:
        raise Exception()
    
    for glb_path in glb_paths:
        # Delete any startup objects
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        for image in bpy.data.images:
            bpy.data.images.remove(image, do_unlink=True)

        try:
            bpy.ops.import_scene.gltf(filepath=glb_path)
        except:
            with open('error_files.txt', 'a') as f:
                f.write(f'{glb_path}\n')
            continue 

        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='SELECT')

        filename = os.path.basename(glb_path).replace('.glb', '')
        filedir = os.path.join(glb_dir, filename)
        os.makedirs(filedir, exist_ok=True)
    
        for im in bpy.data.images:
            if im.name != 'Render Result' and im.name != 'Viewer Node':
                im.save(filepath=os.path.join(filedir, im.name+'.jpg'))

        vertices = [] 
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                mesh = obj.data
                for vertex in mesh.vertices:
                    global_coord = obj.matrix_world @ vertex.co
                    vertices.append(np.array(global_coord))

        vertices = np.stack(vertices)
        x_min, x_max = min(vertices[:, 0]), max(vertices[:, 0])
        y_min, y_max = min(vertices[:, 1]), max(vertices[:, 1])
        z_min, z_max = min(vertices[:, 2]), max(vertices[:, 2])
        # print(x_min, x_max)
        # print(y_min, y_max)
        # print(z_min, z_max)
        translation = -np.array([(x_max+x_min)/2., (y_max+y_min)/2., z_min])
        scale = 4 / (max(x_max-x_min, y_max-y_min, z_max-z_min) + 1e-6)
        translation *= scale
        # print(translation, scale)
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                obj.scale.x *= scale
                obj.scale.y *= scale
                obj.scale.z *= scale
                obj.location.x += translation[0]
                obj.location.y += translation[1]
                obj.location.z += translation[2]
        
        obj_path = os.path.join(filedir, 'scene.obj')
        bpy.ops.export_scene.obj(filepath=obj_path, use_selection=True, use_normals=False, use_materials=False, use_uvs=True, axis_forward='Y', axis_up='Z')

        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                print(obj.name)
                obj_material_nodes = obj.active_material.node_tree.nodes
                albedo, rough = 0.8, np.random.uniform(0.01, 0.9)
                r_ind = np.random.uniform(1.3, 1.7)
                for node in obj_material_nodes:
                    if node.type == 'TEX_IMAGE':
                        print('\t', node.label, node.image.name)                        
                        if 'color' in node.label.lower():
                            save_name = node.image.name+'.jpg'
                            albedo = os.path.abspath(os.path.join(filedir, save_name))
                        elif 'roughness' in node.label.lower():
                            save_name = node.image.name+'.jpg'
                            rough = os.path.abspath(os.path.join(filedir, save_name))
                if not isinstance(albedo, str) and not isinstance(rough, str):
                    with open('obj_files.txt', 'a') as f:
                        f.write(f'{glb_path}\n')
                else:
                    with open('obj_texture_files.txt', 'a') as f:
                        f.write(f'{filedir}\n')
            
                break
        
        
        
        model_dict = load_pplastic_model(albedo, rough, 
            ior=r_ind, model_path=os.path.abspath(obj_path), model_type='obj', dist='ggx')

        with open(os.path.join(filedir, 'scene.json'), 'w') as f:
            json.dump(model_dict, f, indent=4)

        
if __name__ == "__main__":
    # running: blender -b -P glb2obj.py -- <dir to the glb files>
    main()