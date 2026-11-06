# SPDX-License-Identifier: GPL-3.0-or-later
"""Render a model as a 2x2 sheet: front, right, back, three-quarter.

    blender -b --factory-startup --python render_sheet.py -- <in.glb> <out.png> <size>

Prints one line ``RESULT {json}`` with the model's dimensions (metres, glTF
axes: x width, y height, z depth), triangle and material counts.
"""

import json
import math
import sys

import bpy
from mathutils import Vector

source, output, size = sys.argv[sys.argv.index("--") + 1 :][:3]
size = int(size)
half = size // 2

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=source)
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
if not meshes:
    raise SystemExit("no mesh in " + source)
corners = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
lo = Vector([min(c[i] for c in corners) for i in range(3)])
hi = Vector([max(c[i] for c in corners) for i in range(3)])
center, extent = (lo + hi) / 2, max(hi - lo)

scene = bpy.context.scene
engines = [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
scene.render.resolution_x = scene.render.resolution_y = half
scene.view_settings.view_transform = "Standard"
world = bpy.data.worlds.new("Studio")
scene.world = world
world.use_nodes = True
background = world.node_tree.nodes["Background"]
background.inputs[0].default_value = (0.78, 0.78, 0.8, 1)
background.inputs[1].default_value = 1.0
sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
scene.collection.objects.link(sun)
sun.data.energy = 3
sun.rotation_euler = (math.radians(50), 0, math.radians(30))
camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
scene.collection.objects.link(camera)
scene.camera = camera
camera.data.type = "ORTHO"
camera.data.ortho_scale = extent * 1.25
camera.data.clip_end = extent * 10

# Blender is Z-up; glTF's front (+Z) imports as Blender's -Y.
views = [(0, 12), (90, 12), (180, 12), (45, 25)]
pixels = [0.0] * (size * size * 4)
for index, (azimuth, elevation) in enumerate(views):
    a, e = math.radians(azimuth), math.radians(elevation)
    direction = Vector((math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)))
    camera.location = center + direction * extent * 3
    camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = f"{output}.{index}.png"
    bpy.ops.render.render(write_still=True)
    view = bpy.data.images.load(scene.render.filepath)
    data = view.pixels[:]
    ox, oy = (index % 2) * half, (1 - index // 2) * half
    for y in range(half):
        start = ((oy + y) * size + ox) * 4
        pixels[start : start + half * 4] = data[y * half * 4 : (y + 1) * half * 4]
sheet = bpy.data.images.new("Sheet", size, size)
sheet.pixels = pixels
sheet.filepath_raw = output
sheet.file_format = "PNG"
sheet.save()

triangles = 0
for o in meshes:
    o.data.calc_loop_triangles()
    triangles += len(o.data.loop_triangles)
materials = {m.name for o in meshes for m in o.data.materials if m}
width, depth, height = (hi - lo)  # Blender axes: x, y (depth), z (up)
print("RESULT " + json.dumps({
    "dimensions_m": {"width": round(width, 4), "height": round(height, 4), "depth": round(depth, 4)},
    "triangles": triangles,
    "objects": len(meshes),
    "materials": len(materials),
}))
