"""Shared Blender-side helpers.

Imported by ``blender_postprocess.py`` and ``blender_postprocess_multobj.py``.
Both entry scripts share the same setup boilerplate (render engine init,
mesh smoothing + UV + displacement modifiers, principled BSDF wiring,
camera placement); before Phase 4 the two scripts diverged by accident in
several spots. Anything that touched ``bpy``/``mathutils`` lives here.

This module is **only** safe to import inside Blender (it does
``import bpy`` at module load).
"""
from __future__ import annotations

import os
import bpy
import mathutils
import numpy as np


DEFAULT_ALBEDO_PATH = 'assets/materials/default/0.png'
DEFAULT_ROUGH_PATH = 'assets/materials/default/0.png'


# ---------------------------------------------------------------------------
# scene + render engine setup
# ---------------------------------------------------------------------------

def init_render(args, *, extra_view_layers: tuple[str, ...] = ()):
    """Configure render engine, AOV, view layers; return (context, scene, render).

    ``extra_view_layers`` is consumed by the metal+dielec pipeline to spawn
    holdout layers; the single-object pipeline passes ``()``.
    """
    context = bpy.context
    scene = bpy.context.scene
    render = bpy.context.scene.render
    render.engine = args.engine
    render.image_settings.file_format = args.format
    render.image_settings.color_depth = args.color_depth
    render.resolution_percentage = 100
    scene.use_nodes = True
    if render.engine == 'CYCLES':
        scene.cycles.device = 'GPU'

    main_view_layer = scene.view_layers["ViewLayer"]
    main_view_layer.use_pass_z = True
    main_view_layer.use_pass_normal = True
    main_view_layer.use_pass_diffuse_color = True

    bpy.ops.scene.view_layer_add_aov()
    main_view_layer.aovs['AOV'].name = 'Roughness'

    extra_layers = [
        context.scene.view_layers.new(name=name) for name in extra_view_layers
    ]
    return context, scene, render, extra_layers


def clear_scene_objects_and_images():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for image in list(bpy.data.images):
        bpy.data.images.remove(image, do_unlink=True)


# ---------------------------------------------------------------------------
# mesh import + macro shape augmentation
# ---------------------------------------------------------------------------

def _attach_subdivide(obj, level):
    obj.modifiers.new("subsurface", type='SUBSURF')
    obj.modifiers['subsurface'].subdivision_type = 'SIMPLE'
    obj.modifiers['subsurface'].levels = level
    obj.modifiers['subsurface'].use_creases = True
    obj.modifiers['subsurface'].use_custom_normals = True


def _attach_remesh(obj, octree_depth, scale, subsurf_levels):
    obj.modifiers.new("remesh", type='REMESH')
    obj.modifiers['remesh'].mode = 'SMOOTH'
    obj.modifiers['remesh'].octree_depth = octree_depth
    obj.modifiers['remesh'].scale = scale
    obj.modifiers.new("remesh_subsurface", type='SUBSURF')
    obj.modifiers['remesh_subsurface'].subdivision_type = 'CATMULL_CLARK'
    obj.modifiers['remesh_subsurface'].levels = subsurf_levels
    obj.modifiers['remesh_subsurface'].use_creases = True
    obj.modifiers['remesh_subsurface'].use_custom_normals = True


def import_and_smooth_mesh(
    *,
    obj_path,
    obj_name,
    world_mat,
    smooth_level,
    smooth_level_scale,
    vertex_num,
    smooth_cache_path=None,
):
    """Import an OBJ, apply (subsurf | remesh) per ``smooth_level``,
    return the active object."""
    context = bpy.context
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.import_scene.obj(filepath=obj_path)
    obj = context.selected_objects[0]
    obj.name = obj_name
    context.view_layer.objects.active = obj
    obj.matrix_world = mathutils.Matrix(world_mat)

    needs_smooth = smooth_cache_path is None or not os.path.isfile(smooth_cache_path)
    if not needs_smooth:
        return obj

    if smooth_level == 0:
        if vertex_num < 5000:
            subdivide_level = 3
        elif vertex_num < 20000:
            subdivide_level = 2
        elif vertex_num < 170000:
            subdivide_level = 1
        else:
            subdivide_level = 0
        if subdivide_level > 0:
            _attach_subdivide(obj, subdivide_level)
        bpy.ops.object.convert(target='MESH')
        if smooth_cache_path is not None:
            bpy.ops.export_scene.obj(
                filepath=smooth_cache_path, use_selection=True,
                use_normals=False, use_materials=False, use_uvs=False,
                axis_forward='Y', axis_up='Z',
            )
    elif smooth_level == 1:
        _attach_remesh(obj, octree_depth=7, scale=smooth_level_scale, subsurf_levels=2)
        bpy.ops.object.convert(target='MESH')
    elif smooth_level == 2:
        _attach_remesh(obj, octree_depth=6, scale=smooth_level_scale, subsurf_levels=3)
        bpy.ops.object.convert(target='MESH')
    else:
        raise NotImplementedError(f'invalid model smooth level: {smooth_level}')
    return obj


# ---------------------------------------------------------------------------
# UV + displacement + material wiring
# ---------------------------------------------------------------------------

def apply_smart_uv_and_augment(obj, uv_transform, uv_transform_center):
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project()
    bpy.ops.object.mode_set(mode='OBJECT')
    uv_layer = obj.data.uv_layers.active
    uvs = np.empty((2 * len(uv_layer.data), 1), 'f')
    uv_layer.data.foreach_get('uv', uvs)
    S = mathutils.Matrix.Diagonal(uv_transform)
    scaled_uvs = np.dot(uvs.reshape((-1, 2)) - uv_transform_center, S) + uv_transform_center
    uv_layer.data.foreach_set('uv', scaled_uvs.ravel())


def apply_displacement(obj, disp_path, strength, model_idx):
    if disp_path is None:
        return
    disp_img = bpy.data.images.load(disp_path)
    disp_img.name = f'disp_{model_idx}'
    disp_img.colorspace_settings.name = 'Non-Color'
    disp_texture = bpy.data.textures.new('disp', type='IMAGE')
    disp_texture.image = disp_img
    obj.modifiers.new("disp", type='DISPLACE')
    obj.modifiers['disp'].texture = disp_texture
    obj.modifiers['disp'].texture_coords = 'UV'
    obj.modifiers['disp'].strength = strength
    bpy.ops.object.convert(target='MESH')


def _ensure_image(path, *, fallback_name, model_idx, color_space=None):
    is_default = path == DEFAULT_ALBEDO_PATH or path == DEFAULT_ROUGH_PATH
    if is_default and fallback_name in bpy.data.images.keys():
        return bpy.data.images[fallback_name]
    img = bpy.data.images.load(path)
    img.name = fallback_name if is_default else f'{fallback_name.split("_")[0]}_{model_idx}'
    if color_space is not None:
        img.colorspace_settings.name = color_space
    return img


def wire_principled_material(
    obj, *, albedo_path, rough_path, model_idx,
    albedo_color_space=None,
):
    """Hook albedo + roughness textures + a Roughness AOV output."""
    obj = bpy.data.objects[obj.name]
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = obj
    albedo_img = _ensure_image(
        albedo_path, fallback_name='albedo_default',
        model_idx=model_idx, color_space=albedo_color_space,
    )
    rough_img = _ensure_image(
        rough_path, fallback_name='rough_default',
        model_idx=model_idx, color_space='Non-Color',
    )
    nodes = obj.active_material.node_tree.nodes
    links = obj.active_material.node_tree.links
    shader = nodes.get('Principled BSDF')

    node_albedo = nodes.new(type='ShaderNodeTexImage')
    node_albedo.image = albedo_img
    links.new(shader.inputs['Base Color'], node_albedo.outputs['Color'])

    node_rough = nodes.new(type='ShaderNodeTexImage')
    node_rough.image = rough_img
    links.new(shader.inputs['Roughness'], node_rough.outputs['Color'])

    node_rough_aov = nodes.new(type='ShaderNodeOutputAOV')
    node_rough_aov.name = 'Roughness'
    links.new(node_rough_aov.inputs['Color'], node_rough.outputs['Color'])


# ---------------------------------------------------------------------------
# camera + render loop
# ---------------------------------------------------------------------------

# Blender-axes -> Mitsuba-axes flip used everywhere we re-import a camera
# matrix from Mitsuba. ``[-1,1,1,1]`` flips X and Z and matches the historical
# convention.
_BL_AXES_FLIP = np.asarray([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


def build_camera(scene):
    bpy.ops.object.camera_add()
    cam = scene.objects['Camera']
    scene.collection.objects.link(cam)
    bpy.context.view_layer.objects.active = cam
    scene.camera = cam
    return cam


def render_camera_pass(*, cam, scene, render, cam_world_mat, up_axis, fov,
                        resolution, save_path):
    render.resolution_x = resolution[1]
    render.resolution_y = resolution[0]
    cam.matrix_world = mathutils.Matrix(cam_world_mat @ _BL_AXES_FLIP)
    cam.up_axis = up_axis
    cam.data.type = 'PERSP'
    cam.data.lens_unit = 'FOV'
    cam.data.angle = np.deg2rad(fov)
    scene.render.filepath = save_path
    bpy.ops.render.render(write_still=True)


# ---------------------------------------------------------------------------
# collection-holdout (used only by the metal+dielec pipeline)
# ---------------------------------------------------------------------------

def split_into_material_collections(scene, dielec_obj_names, metal_obj_names,
                                    *, view_layer_dielec, view_layer_metal):
    """Move metal and dielec meshes into separate collections so that the
    holdout view layers can mask one material out per pass."""
    collection_dielec = bpy.data.collections.new("dielec")
    scene.collection.children.link(collection_dielec)
    collection_metal = bpy.data.collections.new("metal")
    scene.collection.children.link(collection_metal)
    collection_default = bpy.data.collections.get("Collection")
    for name in dielec_obj_names:
        collection_dielec.objects.link(bpy.data.objects[name])
        collection_default.objects.unlink(bpy.data.objects[name])
    for name in metal_obj_names:
        collection_metal.objects.link(bpy.data.objects[name])
        collection_default.objects.unlink(bpy.data.objects[name])

    for vl, target in ((view_layer_dielec, 'dielec'), (view_layer_metal, 'metal')):
        layer_collections = [vl.layer_collection]
        while layer_collections:
            lc = layer_collections.pop(0)
            if lc.collection.name == target:
                lc.holdout = True
                break
            layer_collections.extend(lc.children)
