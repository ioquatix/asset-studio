# SPDX-License-Identifier: GPL-3.0-or-later
# Run in the live Blender by the asset-studio MCP server, with ARGS defined:
# {"asset_id", "title", "path"}. Imports the GLB into its own collection,
# tagged so pull/close can find it again.
import bpy

collection_name = "AS " + (ARGS["title"] or ARGS["asset_id"])[:48] + " [" + ARGS["asset_id"][:8] + "]"
existing = [c for c in bpy.data.collections if c.get("asset_studio_id") == ARGS["asset_id"]]
if existing:
    collection = existing[0]
    objects = [o.name for o in collection.all_objects]
    result = {"collection": collection.name, "objects": objects, "already_open": True}
else:
    collection = bpy.data.collections.new(collection_name)
    collection["asset_studio_id"] = ARGS["asset_id"]
    bpy.context.scene.collection.children.link(collection)
    layer = bpy.context.view_layer
    previous = layer.active_layer_collection
    layer.active_layer_collection = layer.layer_collection.children[collection.name]
    before = set(bpy.data.objects)
    try:
        bpy.ops.import_scene.gltf(filepath=ARGS["path"])
    finally:
        layer.active_layer_collection = previous
    imported = [o for o in bpy.data.objects if o not in before]
    for o in imported:
        o["asset_studio_id"] = ARGS["asset_id"]
        if o.name not in collection.objects:
            collection.objects.link(o)
    result = {"collection": collection.name, "objects": [o.name for o in imported], "already_open": False}
