import argparse
import copy
import os
import xml.etree.ElementTree as ET
import numpy as np
import trimesh

# from SyntheticArticulatedData.generation.utils import get_cam_params


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


def generate_mujoco_scene_xml(urdf_file, xml_file, obj_type='microwave'):
    # Load xml tree of both files
    urdf_tree = ET.parse(urdf_file)
    urdf_root = urdf_tree.getroot()

    xml_tree = ET.parse(xml_file)
    xml_root = xml_tree.getroot()

    # znear, zfar, fovy = get_cam_params()  # Calibration parameters
    znear, zfar, fovy = 0.1, 12, 85

    xml_root = add_texture(xml_root)
    xml_root = copy_mesh_name_tags(urdf_root, xml_root, obj_type)
    xml_root = update_complier_tag(xml_root)
    xml_root = add_gravity_tag(xml_root, val=[0., 0., 0.])
    xml_root = add_contact_tag(xml_root)
    xml_root = add_statistic_tag(xml_root)
    xml_root = add_global_visual_properties(xml_root, clip_range=[znear, zfar])
    xml_root = add_base_body(xml_root)
    xml_root = add_camera(xml_root, name='external_camera_0', fovy=fovy)
    xml_root = add_actuator_tags(xml_root)

    # save new xml file
    xml_tree.write(xml_file, xml_declaration=True)


def copy_mesh_name_tags(urdf_root, xml_root, obj_type):
    # More elegant solution would be to load from json provided in the dataset
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

                if visElemList[geom.attrib['mesh']] == 'handle':
                    geom.set("material", "geomHandle")

    return xml_root


def add_actuator_tags(xml_root):
    xml_root[-1].tail = "\n\t"
    act = ET.SubElement(xml_root, 'actuator')
    act.tail = "\n"
    act.text = "\n\t\t"

    for i, jnt in enumerate(xml_root.iter("joint")):
        # Add actuator node as a child of root node
        vel = ET.SubElement(act, "velocity")
        vel.set('joint', jnt.attrib['name'])
        vel.set('name', 'act_' + str(i))
        vel.set('kv', '10')
        vel.tail = "\n\t\t"

    vel.tail = "\n\t"  # Correctly format tha last one
    return xml_root


def update_complier_tag(xml_root):
    c_tag = xml_root.find('compiler')
    c_tag.set('eulerseq', 'zxy')
    return xml_root


def add_gravity_tag(xml_root, val):
    xml_root[-1].tail = "\n\t"
    g_tag = ET.SubElement(xml_root, 'option')
    g_tag.set('gravity', '{} {} {}'.format(val[0], val[1], val[2]))
    return xml_root


def add_contact_tag(xml_root):
    xml_root[-1].tail = "\n\t"
    o_tag = ET.SubElement(xml_root, 'option')
    o_tag.tail = "\n"
    o_tag.text = "\n\t\t"
    c_tag = ET.SubElement(o_tag, 'flag')
    c_tag.set('contact', 'disable')
    c_tag.tail = "\n\t"
    return xml_root


def add_statistic_tag(xml_root, bb_center=[0., 0., 0.]):
    xml_root[-1].tail = "\n\t"
    s_tag = ET.SubElement(xml_root, 'statistic')
    s_tag.set('extent', '1.0')
    s_tag.set('center', '{} {} {}'.format(bb_center[0], bb_center[1], bb_center[2]))
    return xml_root


def add_global_visual_properties(xml_root, clip_range=[0.1, 12.0]):
    xml_root[-1].tail = "\n\t"
    v_tag = ET.SubElement(xml_root, 'visual')
    v_tag.tail = "\n"
    v_tag.text = "\n\t\t"
    m_tag = ET.SubElement(v_tag, 'map')
    m_tag.set('force', '0.1')
    m_tag.set('fogstart', '3.')
    m_tag.set('fogend', '5.')
    m_tag.set('znear', str(clip_range[0]))
    m_tag.set('zfar', str(clip_range[1]))
    m_tag.tail = "\n\t"
    return xml_root


def add_texture(xml_root):
    asset = xml_root.find('asset')
    asset.text = "\n\t\t"
    asset.tail = "\n\t"
    asset[-1].tail = "\n\t\t"
    t1 = ET.SubElement(asset, 'texture')
    t1.set('builtin', 'flat')
    t1.set('name', 'objTex')
    t1.set('height', '32')
    t1.set('width', '32')
    t1.set('rgb1', '1 1 1')
    t1.set('type', 'cube')
    t1.tail = "\n\t\t"
    t2 = ET.SubElement(asset, 'texture')
    t2.set('builtin', 'flat')
    t2.set('name', 'handleTex')
    t2.set('height', '32')
    t2.set('width', '32')
    t2.set('rgb1', '0.8 0.8 0.8')
    t2.set('type', 'cube')
    t2.tail = "\n\t\t"

    m1 = ET.SubElement(asset, 'material')
    m1.set('name', 'geomObj')
    m1.set('shininess', '0.03')
    m1.set('specular', '0.75')
    m1.set('texture', 'objTex')
    m1.tail = '\n\t\t'
    m2 = ET.SubElement(asset, 'material')
    m2.set('name', 'geomHandle')
    m2.set('shininess', '0.03')
    m2.set('specular', '0.75')
    m2.set('texture', 'handleTex')
    m2.tail = '\n\t'

    for geom in xml_root.iter('geom'):
        geom.set('material', 'geomObj')
    return xml_root


def add_base_body(xml_root, name="base", pose=[0, 0, 0], ori=[1., 0., 0., 0.]):
    xml_root[-1].tail = "\n\t\t"
    body_base = ET.SubElement(xml_root, 'body')
    body_base.text = "\n\t\t\t"
    body_base.set('name', name)
    body_base.set('pos', '{} {} {}'.format(pose[0], pose[1], pose[2]))
    # NOTE: orientation quaternion is in wxyz format
    body_base.set('quat', '{} {} {} {}'.format(ori[0], ori[1], ori[2], ori[3]))
    body_base.tail = "\n\t\t"

    world = xml_root.find('worldbody')
    body_base.extend(world)
    world.clear()
    world.append(body_base)
    xml_root.remove(body_base)

    # Formatting
    for b in body_base.findall('geom'):
        b.tail = "\n\t\t\t"
    for b in body_base.findall('body'):
        b.text = "\n\t\t\t\t"
        b.tail = "\n\t\t\t"
        for child in b:
            child.tail = "\n\t\t\t\t"
        child.tail = "\n\t\t\t"
    b.tail = "\n\t\t"
    world.tail = "\n\t"
    world.text = "\n\t\t"
    return xml_root


def add_camera(xml_root, name="cam", pose=[0., 0., 0.], ori=[1., 0., 0., 0.], fovy=85):
    xml_root[-1].tail = "\n\t\t"
    body = ET.SubElement(xml_root.find('worldbody'), 'body')
    body.text = "\n\t\t\t"
    body.set('name', name + '_body')
    body.set('pos', '{} {} {}'.format(pose[0], pose[1], pose[2]))
    body.set('quat', '{} {} {} {}'.format(ori[0], ori[1], ori[2], ori[3]))
    body.tail = "\n\t"
    cam = ET.SubElement(body, 'camera')
    cam.set('euler', '-1.57 1.57 0.0')
    cam.set('fovy', str(fovy))
    cam.set('name', name)
    cam.set('pos', "0. 0. 0.")
    cam.tail = "\n\t\t\t"
    iner = ET.SubElement(body, 'inertial')
    iner.set('pos', '0.0 0.0 0.0')
    iner.set('mass', '1')
    iner.set('diaginertia', '1 1 1')
    iner.tail = "\n\t\t\t"
    jnt = ET.SubElement(body, 'joint')
    jnt.set('name', name+'_jnt')
    jnt.set('pos', '0. 0. 0.')
    jnt.set('axis', '1 0 0')
    jnt.set('type', 'free')
    jnt.tail = "\n\t\t"
    xml_root[-1].tail = "\n\t"
    return xml_root


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
        generate_mujoco_scene_xml(urdf_file=args.input_files[0], xml_file=args.output_files[0], obj_type=args.obj_type)

    else:
        print("Not implemented yet!")
