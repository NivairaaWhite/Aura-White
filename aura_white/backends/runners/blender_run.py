"""Runs *inside Blender's own Python* (`blender --background --python blender_run.py -- <args>`).

Never imported by Aura White directly - only ever executed as a subprocess, exactly like the Pixal3D / TRELLIS.2
runners, so a Blender version difference can never break anything outside this one process. All API calls below
are the standard, documented bpy operators (glTF import/export, Cycles bake, QuadriFlow remesh, voxel remesh) - see
aura_white/backends/blender_tools.py for what calls this and why.

Tasks (--task):
  ao_bake       bake ambient occlusion into Aura White's own UV layout -> a single-channel PNG
  quad_remesh   QuadriFlow quad-dominant retopology to ~target-faces quads, triangulated on export -> GLB
  voxel_remesh  OpenVDB voxel remesh at the given voxel size (closes small holes, unifies topology) -> GLB
"""
import argparse
import sys

import bpy


def _args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["ao_bake", "quad_remesh", "voxel_remesh"])
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--target-faces", type=int, default=20000)
    ap.add_argument("--voxel-size", type=float, default=0.01)
    ap.add_argument("--preserve-boundary", action="store_true")
    return ap.parse_args(argv)


def _clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.images):
        for block in list(coll):
            if block.users == 0:
                coll.remove(block)


def _import_mesh(path: str):
    _clear_scene()
    bpy.ops.import_scene.gltf(filepath=path)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("the imported GLB has no mesh objects")
    obj = meshes[0]
    bpy.context.view_layer.objects.active = obj
    for o in meshes:
        o.select_set(o is obj)
    return obj


def _export_mesh(obj, path: str):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.export_scene.gltf(filepath=path, export_format="GLB", use_selection=True,
                              export_yup=True, export_apply=True)


def cmd_ao_bake(a):
    obj = _import_mesh(a.inp)
    if not obj.data.uv_layers:
        raise RuntimeError("the input mesh has no UV map to bake into")
    obj.data.uv_layers[0].active = True
    img = bpy.data.images.new("aura_ao_bake", width=a.size, height=a.size, alpha=False, float_buffer=False)
    mat = bpy.data.materials.new("aura_ao_bake_material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    tex_node = nodes.new("ShaderNodeTexImage")
    tex_node.image = img
    nodes.active = tex_node
    obj.data.materials.clear()
    obj.data.materials.append(mat)

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    try:
        scene.cycles.device = "GPU"
    except Exception:
        pass
    scene.cycles.samples = int(a.samples)
    scene.render.bake.margin = 8
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.bake(type="AO")

    img.filepath_raw = a.out
    img.file_format = "PNG"
    img.save()


def cmd_quad_remesh(a):
    obj = _import_mesh(a.inp)
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.quadriflow_remesh(target_faces=int(a.target_faces), use_mesh_symmetry=False,
                                     use_preserve_boundary=bool(a.preserve_boundary), use_mesh_curvature=True,
                                     mode="FACES")
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.quads_convert_to_tris(quad_method="BEAUTY", ngon_method="BEAUTY")
    bpy.ops.object.mode_set(mode="OBJECT")
    _export_mesh(obj, a.out)


def cmd_voxel_remesh(a):
    obj = _import_mesh(a.inp)
    obj.data.remesh_voxel_size = float(a.voxel_size)
    obj.data.remesh_voxel_adaptivity = 0.0
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.voxel_remesh()
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.quads_convert_to_tris(quad_method="BEAUTY", ngon_method="BEAUTY")
    bpy.ops.object.mode_set(mode="OBJECT")
    _export_mesh(obj, a.out)


def main():
    a = _args()
    {"ao_bake": cmd_ao_bake, "quad_remesh": cmd_quad_remesh, "voxel_remesh": cmd_voxel_remesh}[a.task](a)
    print(f"aura_white.blender_run: {a.task} -> {a.out}")


if __name__ == "__main__":
    main()
