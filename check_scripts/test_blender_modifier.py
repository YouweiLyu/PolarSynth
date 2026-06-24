# Example:
# blender -b -P check_scripts/test_blender_modifier.py

import os
import bpy
import mathutils

def test_subdivide_modifier(subdivide_level=1):
    obj_model_path = 'assets/3D_assets/models/cgtrader/hero_warrior.obj'
    displace_path = 'assets/materials/forest_ground_04/forest_ground_04_4K_Displacement.jpg'

    context = bpy.context
    scene = bpy.context.scene
    render = bpy.context.scene.render

    render.engine = 'CYCLES'
    render.image_settings.file_format = 'OPEN_EXR_MULTILAYER' # ('PNG', 'OPEN_EXR', 'JPEG', ...)
    render.image_settings.color_depth = '32' # ('8', '16')
    render.resolution_percentage = 100
    render.use_stamp = True
    render.use_stamp_lens = True

    scene.use_nodes = True
    scene.view_layers["ViewLayer"].use_pass_z = True
    scene.view_layers["ViewLayer"].use_pass_normal = True
    scene.view_layers["ViewLayer"].use_pass_diffuse_color = True
    if render.engine == 'CYCLES':
        scene.cycles.device = 'GPU'

    # Delete default cube
    context.active_object.select_set(True)
    bpy.ops.object.delete()
    # Import textured mesh
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.import_scene.obj(filepath=obj_model_path)

    obj = context.selected_objects[0]
    context.view_layer.objects.active = obj
    obj.matrix_world = mathutils.Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

    obj_data = obj.data
    obj_data_dict = {
        'vertices': len(obj_data.vertices),
        'edges': len(obj_data.edges),
        'facets': len(obj_data.polygons),
    }
    print('#### Before Subdivide Modification ####')
    for key in obj_data_dict:
        print(f'\t{key:<8s}: {obj_data_dict[key]:>10d}')
    print('#######################################')

    # bpy.ops.image.open(filepath=albedo_path)
    bpy.ops.image.open(filepath=displace_path)
    # bpy.data.images[os.path.split(albedo_path)[-1]].name = 'color'
    bpy.data.images[os.path.split(displace_path)[-1]].name = 'displace'
    bpy.data.images['displace'].colorspace_settings.name = 'Non-Color'

    obj.modifiers.new("subsurface", type='SUBSURF')
    obj.modifiers['subsurface'].subdivision_type = 'SIMPLE'
    obj.modifiers['subsurface'].levels = subdivide_level
    obj.modifiers['subsurface'].use_creases = True
    obj.modifiers['subsurface'].use_custom_normals = True

    displace_map = bpy.data.images['displace']
    obj.modifiers.new("displace", type='DISPLACE')
    displace_tex = bpy.data.textures.new('displace', type='IMAGE')
    displace_tex.image = displace_map
    obj.modifiers['displace'].texture = displace_tex
    obj.modifiers['displace'].texture_coords = 'UV'
    obj.modifiers['displace'].strength = 0.1

    obj = context.selected_objects[0]
    obj_data = obj.data
    bpy.ops.object.convert(target='MESH')
    obj_data_dict = {
        'vertices': len(obj_data.vertices),
        'edges': len(obj_data.edges),
        'facets': len(obj_data.polygons),
    }
    print('\n#### After Subdivide Modification ####')
    for key in obj_data_dict:
        print(f'\t{key:<8s}: {obj_data_dict[key]:>10d}')
    print('######################################')
    # Place camera
    # camera_data = bpy.data.cameras.new('camera')
    # cam = bpy.data.objects.new('camera', camera_data)
    # cam = scene.objects['Camera']
    # context.view_layer.objects.active = cam
    # cam_world_mat = meta_info['camera_mat'] @ np.asarray([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
    # cam.matrix_world = mathutils.Matrix(cam_world_mat)
    # cam.up_axis = meta_info['camera_up_axis']
    # cam.data.type = 'PERSP'
    # cam.data.lens_unit = 'FOV'
    # cam.data.angle = np.deg2rad(meta_info['fov'])
    # scene.collection.objects.link(cam)

    # Place extra light 
    # light_world_mat = meta_info['dist_light_info'][0]['mat'] @ np.asarray([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
    # light_data = bpy.data.lights.new(name="light_data", type='SUN')
    # light_data.energy = meta_info['dist_light_info'][0]['int']
    # light = bpy.data.objects.new(name="light", object_data=light_data)
    # light.matrix_world = mathutils.Matrix(light_world_mat)
    # scene.collection.objects.link(light)

    # render_file_path = os.path.join(args.save_dir, os.path.split(obj_model_path)[-1].replace(".obj", "_blender.exr"))
    # scene.render.filepath = render_file_path
    # bpy.ops.render.render(write_still=True)
    # model_save_path = os.path.join(args.save_dir, 'tmp.obj')
    # bpy.ops.export_scene.obj(filepath=model_save_path, use_normals=False, use_materials=False, axis_forward='Y', axis_up='Z')

if __name__ == '__main__':
    test_subdivide_modifier(1)