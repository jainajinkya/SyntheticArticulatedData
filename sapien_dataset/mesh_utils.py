import numpy as np
import trimesh
import copy
import argparse
import os


def make_mesh_watertight(file_in, file_out):
    mesh = trimesh.load_mesh(file_in)
    bnds = np.array(mesh.bounding_box.extents)
    bad_cols = np.nonzero(bnds < 1e-5)
    if bad_cols[0].size > 0:
        # new_vert = copy.copy(mesh.vertices)
        # for k in bad_cols:
        #     new_vert[:, k[0]] += np.random.uniform(low=1e-6, high=2e-6, size=len(mesh.vertices))
        # mesh2 = trimesh.convex.convex_hull(new_vert)

        mesh2 = trimesh.creation.extrude_triangulation(np.delete(mesh.vertices, bad_cols, axis=1),
                                                       mesh.faces, height=0.001)
        mesh2 = trimesh.convex.convex_hull(mesh2.vertices)
    else:
        mesh2 = copy.copy(mesh)
    with open(file_out, 'wb') as f:
        trimesh.exchange.export.export_mesh(mesh2, file_obj=f, file_type='stl')


def make_mesh_watertight_obj(file_in, file_out):
    try:
        mesh = trimesh.load(file_in)
        mesh2 = copy.copy(mesh)
        if not isinstance(mesh, trimesh.Scene):
            bnds = np.array(mesh.bounding_box.extents)
            bad_cols = np.nonzero(bnds < 1e-5)
            if bad_cols[0].size > 0:
                new_vert = copy.copy(mesh.vertices)
                for k in bad_cols:
                    new_vert[:, k[0]] += np.random.uniform(low=1e-6, high=2e-6, size=len(mesh.vertices))
                mesh2 = trimesh.convex.convex_hull(new_vert)

        with open(file_out, 'wb') as f:
            trimesh.exchange.export.export_mesh(mesh2, file_obj=f, file_type='stl')
    except TypeError:
        raise TypeError("Will need to use externally generated .stl file for this")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_mesh', type=str, nargs='+', help='input mesh files')
    parser.add_argument('-o', '--output_mesh', type=str, nargs='+', help='input mesh files')

    args = parser.parse_args()
    if args.output_mesh is None:
        for mesh_in in args.input_mesh:
            # mesh_out = mesh_in.replace('.stl', '-corrected.stl')
            mesh_in = os.path.abspath(mesh_in)
            mesh_out = copy.copy(mesh_in)
            make_mesh_watertight(mesh_in, mesh_out)
            print("Watertight meshes created for mesh:{} and saved in:{}".format(mesh_in, mesh_out))
    else:
        for mesh_in, mesh_out in zip(args.input_mesh, args.output_mesh):
            mesh_in = os.path.abspath(mesh_in)
            mesh_out = os.path.abspath(mesh_out)
            make_mesh_watertight(mesh_in, mesh_out)
            print("Watertight meshes created for mesh:{} and saved in:{}".format(mesh_in, mesh_out))
