import argparse
import copy
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

import numpy as np
import trimesh


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


def copy_mesh_name_tags(urdf_file, xml_file, obj_type='microwave'):
    # Load xml tree of both files
    urdf_tree = ET.parse(urdf_file)
    urdf_root = urdf_tree.getroot()

    xml_tree = ET.parse(xml_file)
    xml_root = xml_tree.getroot()

    # More elegnat solution load from json provided in the dataset
    geom_names = []
    if obj_type == 'microwave':
        geom_names = ['body', 'frame', 'door', 'glass', 'handle', 'tray']

    # find the name in the urdf file
    visElemList = {}
    for vis in urdf_root.iter("visual"):
        m_name = vis.find('geometry').find('mesh').attrib['filename'].replace('textured_objs/', '').replace('.stl', '')
        text_name = vis.attrib['name']

        if np.size(np.where([x in text_name for x in geom_names])[0]) > 0:
            text_name = geom_names[np.where([x in text_name for x in geom_names])[0][0]]
            visElemList[m_name] = text_name

    # find and update the tag in the xml file
    for body in xml_root.iter('body'):
        for geom in body.findall('geom'):
            if geom.attrib['mesh'] in visElemList.keys():
                geom.set("name", visElemList[geom.attrib['mesh']])

    # save new xml file
    xml_tree.write(xml_file, xml_declaration=True)


def add_actuator_tags(xml_file):
    xml_tree = ET.parse(xml_file)
    xml_root = xml_tree.getroot()

    for i, jnt in enumerate(xml_root.iter("joint")):
        xml_root[-1].tail = "\n\t"

        # Add actuator node as a child of root node
        act = ET.SubElement(xml_root, 'actuator')
        vel = ET.SubElement(act, "velocity")
        vel.set('joint', jnt.attrib['name'])
        vel.set('name', 'act_' + str(i))
        vel.set('kv', '10')

        # Properly formatting
        act.tail = "\n"
        act.text = "\n\t\t"
        vel.tail = "\n\t"

    xml_tree.write(xml_file, xml_declaration=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_files', type=str, nargs='+', help='input files')
    parser.add_argument('-o', '--output_files', type=str, nargs='+', help='onput files')
    parser.add_argument('-cm', '--correct-mesh', action='store_true', help='correct meshes')
    parser.add_argument('-uxt', '--update-xml-tags', action='store_true', help='copy geom body names in xml')
    parser.add_argument('--obj-type', type=str, default='microwave', help='object category')
    args = parser.parse_args()

    if args.correct_mesh:
        if args.output_files is None:
            args.output_files = copy.copy(args.input_files)   # Update input mesh file inplace

        for mesh_in, mesh_out in zip(args.input_files, args.output_files):
            mesh_in = os.path.abspath(mesh_in)
            mesh_out = os.path.abspath(mesh_out)
            make_mesh_watertight(mesh_in, mesh_out)
            print("Watertight meshes created for mesh:{} and saved in:{}".format(mesh_in, mesh_out))

    elif args.update_xml_tags:
        copy_mesh_name_tags(urdf_file=args.input_files[0], xml_file=args.output_files[0], obj_type=args.obj_type)
        add_actuator_tags(xml_file=args.output_files[0])

    else:
        print("Not implemented yet!")
