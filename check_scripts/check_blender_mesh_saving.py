# Example:
# blender -b -P render_normal_blender_from_mitsuba.py -- renderings/polar_scenes/sgl_obj/sphere_0_meta.npy

import argparse, sys, os
import numpy as np
import bpy
import mathutils

parser = argparse.ArgumentParser(description='Renders normal maps given rendering meta information.')
parser.add_argument('meta_file_path', type=str,
                    help='Path to the meta information file.')
parser.add_argument('--save_dir', type=str, default='renderings_blender/polar_scenes',
                    help='The path the output will be saved to.')
parser.add_argument('--remove_doubles', type=bool, default=True,
                    help='Remove double vertices to improve mesh quality.')
parser.add_argument('--edge_split', type=bool, default=True,
                    help='Adds edge split filter.')
parser.add_argument('--color_depth', type=str, default='32',
                    help='Number of bit per channel used for output. Either 8 or 16.')
parser.add_argument('--format', type=str, default='OPEN_EXR_MULTILAYER',
                    help='Format of files generated. Either PNG, OPEN_EXR, or OPEN_EXR_MULTILAYER')
parser.add_argument('--engine', type=str, default='CYCLES',
                    help='Blender internal engine for rendering. E.g. CYCLES, BLENDER_EEVEE, ...')

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)

# Set up rendering


assert (os.path.exists(args.meta_file_path) and args.meta_file_path.endswith('meta.npy')), \
    f"Directory '{args.meta_file_path}' doesn't exsit."
os.makedirs(args.save_dir, exist_ok=True)

meta_info = np.load(args.meta_file_path, allow_pickle=True).item()

model_path = 'renderings_blender/polar_scenes/asian_dragon_0.obj'
normal_map_path = 'assets/materials/forest_ground_04/forest_ground_04_4K_NormalGL.jpg'
albedo_path = 'assets/materials/forest_ground_04/forest_ground_04_4K_Color.jpg'

context = bpy.context
scene = bpy.context.scene
render = bpy.context.scene.render

render.engine = args.engine
render.image_settings.file_format = args.format # ('PNG', 'OPEN_EXR', 'JPEG, ...)
render.image_settings.color_depth = args.color_depth # ('8', '16')
render.resolution_percentage = 100
resolutions = meta_info['resolutions']
render.resolution_x = resolutions[1]
render.resolution_y = resolutions[0]
render.use_stamp = True
render.use_stamp_lens = True
# render.image_settings.color_mode = 'RGBA' # ('RGB', 'RGBA', ...)
# render.film_transparent = True

scene.use_nodes = True
scene.view_layers["ViewLayer"].use_pass_z = True
scene.view_layers["ViewLayer"].use_pass_normal = True
scene.view_layers["ViewLayer"].use_pass_diffuse_color = True

# Delete default cube
context.active_object.select_set(True)
bpy.ops.object.delete()

# Import textured mesh
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.import_scene.obj(filepath=model_path)

obj = bpy.context.selected_objects[0]
context.view_layer.objects.active = obj
# print(meta_info['model_mats'][0]); exit()
obj.matrix_world = mathutils.Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],])

bpy.ops.image.open(filepath=meta_info['albedo_paths'][0])
bpy.ops.image.open(filepath=meta_info['normal_paths'][0])
bpy.ops.image.open(filepath=meta_info['displace_paths'][0])
bpy.data.images[os.path.split(meta_info['displace_paths'][0])[-1]].name = 'displace'
bpy.data.images[os.path.split(meta_info['albedo_paths'][0])[-1]].name = 'color'
bpy.data.images[os.path.split(meta_info['normal_paths'][0])[-1]].name = 'normal'
bpy.data.images['displace'].colorspace_settings.name = 'Non-Color'
bpy.data.images['normal'].colorspace_settings.name = 'Non-Color'

# obj.modifiers.new("subsurface", type='SUBSURF')
# obj.modifiers['subsurface'].subdivision_type = 'SIMPLE'
# obj.modifiers['subsurface'].levels = 6
# obj.modifiers['subsurface'].render_levels = 3
# # obj.modifiers['subsurface'].use_custom_normals = True

# displace_map = bpy.data.images['displace']
# obj.modifiers.new("displace", type='DISPLACE')
# displace_tex = bpy.data.textures.new('displace', type='IMAGE')
# displace_tex.image = displace_map
# obj.modifiers['displace'].texture = displace_tex
# obj.modifiers['displace'].texture_coords = 'UV'
# obj.modifiers['displace'].strength = 0.2

albedo_map = bpy.data.images['color']
obj_material_nodes = obj.active_material.node_tree.nodes
node_color = obj_material_nodes.new(type='ShaderNodeTexImage')
node_color.image = albedo_map
obj_material_links = obj.active_material.node_tree.links
obj_shader = obj_material_nodes.get('Principled BSDF')
obj_material_links.new(obj_shader.inputs["Base Color"], node_color.outputs["Color"])

normal_map = bpy.data.images['normal']
node_normal_map = obj_material_nodes.new(type='ShaderNodeTexImage')
node_normal_map.image = normal_map
node_normal = obj_material_nodes.new(type='ShaderNodeNormalMap')
node_normal.space = 'TANGENT'
node_normal.uv_map = 'UV_MAP'
obj_material_links.new(node_normal.inputs["Color"], node_normal_map.outputs["Color"])
obj_material_links.new(obj_shader.inputs["Normal"], node_normal.outputs["Normal"])

# Place camera
# camera_data = bpy.data.cameras.new('camera')
# cam = bpy.data.objects.new('camera', camera_data)
cam = scene.objects['Camera']
context.view_layer.objects.active = cam
cam_world_mat = meta_info['camera_mat'] @ np.asarray([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
cam.matrix_world = mathutils.Matrix(cam_world_mat)
cam.up_axis = meta_info['camera_up_axis']
cam.data.type = 'PERSP'
cam.data.lens_unit = 'FOV'
cam.data.angle = np.deg2rad(meta_info['fov'])
scene.collection.objects.link(cam)

render_file_path = os.path.join(args.save_dir, os.path.split(model_path)[-1].replace(".obj", "_check_blender.exr"))
scene.render.filepath = render_file_path
bpy.ops.render.render(write_still=True)
print(obj.matrix_world)
# bpy.ops.export_scene.obj(filepath=render_file_path.replace('_blender.exr', '.obj'), use_normals=False, use_materials=False)