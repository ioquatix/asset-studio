# SPDX-License-Identifier: GPL-3.0-or-later
# Run in the live Blender by the asset-studio MCP server, with ARGS defined:
# {"asset_id", "path"}. Exports the asset's collection (as edited) to a GLB.
import bpy

matches = [c for c in bpy.data.collections if c.get("asset_studio_id") == ARGS["asset_id"]]
if not matches:
    raise RuntimeError("asset %s is not open in Blender; use open_in_blender first" % ARGS["asset_id"])
collection = matches[0]
objects = [o for o in collection.all_objects if o.type in {"MESH", "ARMATURE", "EMPTY", "CURVE"}]
if not objects:
    raise RuntimeError("collection %r has no objects to export" % collection.name)
if bpy.context.object and bpy.context.object.mode != "OBJECT":
    bpy.ops.object.mode_set(mode="OBJECT")
selected = list(bpy.context.selected_objects)
active = bpy.context.view_layer.objects.active
for o in selected:
    o.select_set(False)
for o in objects:
    o.select_set(True)
try:
    bpy.ops.export_scene.gltf(filepath=ARGS["path"], export_format="GLB", use_selection=True,
                              export_apply=True, export_yup=True)
finally:
    for o in objects:
        o.select_set(False)
    for o in selected:
        o.select_set(True)
    bpy.context.view_layer.objects.active = active
result = {"collection": collection.name, "objects": [o.name for o in objects], "path": ARGS["path"]}
