# SPDX-License-Identifier: GPL-3.0-or-later
# Run in the live Blender by the asset-studio MCP server, with ARGS defined:
# {"asset_id"}. Removes the asset's collection and its objects from the scene.
import bpy

removed = []
for collection in [c for c in bpy.data.collections if c.get("asset_studio_id") == ARGS["asset_id"]]:
    for o in list(collection.all_objects):
        bpy.data.objects.remove(o, do_unlink=True)
    removed.append(collection.name)
    bpy.data.collections.remove(collection)
result = {"closed": removed}
